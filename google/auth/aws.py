# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AWS Credentials and AWS Signature V4 Request Signer.

This module provides credentials to access Google Cloud resources from Amazon
Web Services (AWS) workloads. These credentials are recommended over the
use of service account credentials in AWS as they do not involve the management
of long-live service account private keys.

AWS Credentials are initialized using external_account arguments which are
typically loaded from the external credentials JSON file.

This module also provides a definition for an abstract AWS security credentials supplier.
This supplier can be implemented to return valid AWS security credentials and an AWS region
and used to create AWS credentials. The credentials will then call the
supplier instead of using pre-defined methods such as calling the EC2 metadata endpoints.

This module also provides a basic implementation of the
`AWS Signature Version 4`_ request signing algorithm.

AWS Credentials use serialized signed requests to the
`AWS STS GetCallerIdentity`_ API that can be exchanged for Google access tokens
via the GCP STS endpoint.

.. _AWS Signature Version 4: https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html
.. _AWS STS GetCallerIdentity: https://docs.aws.amazon.com/STS/latest/APIReference/API_GetCallerIdentity.html
"""

import abc
from dataclasses import dataclass
import hashlib
import hmac
import http.client as http_client
import ipaddress
import json
import os
import posixpath
import re
from typing import Optional
import urllib
from urllib.parse import urljoin

from google.auth import _helpers
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import external_account

# AWS Signature Version 4 signing algorithm identifier.
_AWS_ALGORITHM = "AWS4-HMAC-SHA256"
# The termination string for the AWS credential scope value as defined in
# https://docs.aws.amazon.com/general/latest/gr/sigv4-create-string-to-sign.html
_AWS_REQUEST_TYPE = "aws4_request"
# The AWS authorization header name for the security session token if available.
_AWS_SECURITY_TOKEN_HEADER = "x-amz-security-token"
# The AWS authorization header name for the auto-generated date.
_AWS_DATE_HEADER = "x-amz-date"
# The default AWS regional credential verification URL.
_DEFAULT_AWS_REGIONAL_CREDENTIAL_VERIFICATION_URL = (
    "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15"
)
# IMDSV2 session token lifetime. This is set to a low value because the session token is used immediately.
_IMDSV2_SESSION_TOKEN_TTL_SECONDS = "300"
# The Amazon ECS container credential provider endpoint. ECS tasks -- including
# tasks on the Fargate launch type, which have no route to the EC2 instance
# metadata server -- retrieve their task role credentials from this link-local
# address at the path the ECS agent injects into the container environment.
# https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-iam-roles.html
_ECS_CONTAINER_CREDENTIALS_HOST = "http://169.254.170.2"
# Environment variable holding the path of the ECS container credential
# provider for the current task. Injected by the ECS agent into every task
# container that has a task role, on both the EC2 and Fargate launch types.
_ECS_CONTAINER_CREDENTIALS_RELATIVE_URI_ENV = "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"
# Environment variables holding the base URI of the ECS task metadata endpoint.
# The v4 variable is injected by current ECS agents; the unversioned one is the
# older v3 endpoint.
_ECS_CONTAINER_METADATA_URI_V4_ENV = "ECS_CONTAINER_METADATA_URI_V4"
_ECS_CONTAINER_METADATA_URI_ENV = "ECS_CONTAINER_METADATA_URI"


def _is_link_local(url):
    """Returns whether a URL addresses a link-local host.

    Used to bound URLs that are read from the environment. Every AWS metadata
    endpoint a workload can reach is a link-local IP literal: the EC2 instance
    metadata server at ``169.254.169.254`` and the ECS task metadata and
    container credential endpoints at ``169.254.170.2``. Nothing that serves
    this data lives on loopback, so a value pointing anywhere else -- loopback
    and ``localhost`` included -- is not a metadata endpoint and is not
    contacted.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL host is a link-local IP literal.
    """
    hostname = urllib.parse.urlsplit(url).hostname
    if not hostname:
        return False
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        # Not an IP literal, so not one of the metadata endpoints.
        return False

    return address.is_link_local


class RequestSigner(object):
    """Implements an AWS request signer based on the AWS Signature Version 4 signing
    process.
    https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html
    """

    def __init__(self, region_name):
        """Instantiates an AWS request signer used to compute authenticated signed
        requests to AWS APIs based on the AWS Signature Version 4 signing process.

        Args:
            region_name (str): The AWS region to use.
        """

        self._region_name = region_name

    def get_request_options(
        self,
        aws_security_credentials,
        url,
        method,
        request_payload="",
        additional_headers={},
    ):
        """Generates the signed request for the provided HTTP request for calling
        an AWS API. This follows the steps described at:
        https://docs.aws.amazon.com/general/latest/gr/sigv4_signing.html

        Args:
            aws_security_credentials (AWSSecurityCredentials): The AWS security credentials.
            url (str): The AWS service URL containing the canonical URI and
                query string.
            method (str): The HTTP method used to call this API.
            request_payload (Optional[str]): The optional request payload if
                available.
            additional_headers (Optional[Mapping[str, str]]): The optional
                additional headers needed for the requested AWS API.

        Returns:
            Mapping[str, str]: The AWS signed request dictionary object.
        """

        additional_headers = additional_headers or {}

        uri = urllib.parse.urlparse(url)
        # Normalize the URL path. This is needed for the canonical_uri.
        # os.path.normpath can't be used since it normalizes "/" paths
        # to "\\" in Windows OS.
        normalized_uri = urllib.parse.urlparse(
            urljoin(url, posixpath.normpath(uri.path))
        )
        # Validate provided URL.
        if not uri.hostname or uri.scheme != "https":
            raise exceptions.InvalidResource("Invalid AWS service URL")

        header_map = _generate_authentication_header_map(
            host=uri.hostname,
            canonical_uri=normalized_uri.path or "/",
            canonical_querystring=_get_canonical_querystring(uri.query),
            method=method,
            region=self._region_name,
            aws_security_credentials=aws_security_credentials,
            request_payload=request_payload,
            additional_headers=additional_headers,
        )
        headers = {
            "Authorization": header_map.get("authorization_header"),
            "host": uri.hostname,
        }
        # Add x-amz-date if available.
        if "amz_date" in header_map:
            headers[_AWS_DATE_HEADER] = header_map.get("amz_date")
        # Append additional optional headers, eg. X-Amz-Target, Content-Type, etc.
        for key in additional_headers:
            headers[key] = additional_headers[key]

        # Add session token if available.
        if aws_security_credentials.session_token is not None:
            headers[_AWS_SECURITY_TOKEN_HEADER] = aws_security_credentials.session_token

        signed_request = {"url": url, "method": method, "headers": headers}
        if request_payload:
            signed_request["data"] = request_payload
        return signed_request


def _get_canonical_querystring(query):
    """Generates the canonical query string given a raw query string.
    Logic is based on
    https://docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html

    Args:
        query (str): The raw query string.

    Returns:
        str: The canonical query string.
    """
    # Parse raw query string.
    querystring = urllib.parse.parse_qs(query)
    querystring_encoded_map = {}
    for key in querystring:
        quote_key = urllib.parse.quote(key, safe="-_.~")
        # URI encode key.
        querystring_encoded_map[quote_key] = []
        for item in querystring[key]:
            # For each key, URI encode all values for that key.
            querystring_encoded_map[quote_key].append(
                urllib.parse.quote(item, safe="-_.~")
            )
        # Sort values for each key.
        querystring_encoded_map[quote_key].sort()
    # Sort keys.
    sorted_keys = list(querystring_encoded_map.keys())
    sorted_keys.sort()
    # Reconstruct the query string. Preserve keys with multiple values.
    querystring_encoded_pairs = []
    for key in sorted_keys:
        for item in querystring_encoded_map[key]:
            querystring_encoded_pairs.append("{}={}".format(key, item))
    return "&".join(querystring_encoded_pairs)


def _sign(key, msg):
    """Creates the HMAC-SHA256 hash of the provided message using the provided
    key.

    Args:
        key (str): The HMAC-SHA256 key to use.
        msg (str): The message to hash.

    Returns:
        str: The computed hash bytes.
    """
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signing_key(key, date_stamp, region_name, service_name):
    """Calculates the signing key used to calculate the signature for
    AWS Signature Version 4 based on:
    https://docs.aws.amazon.com/general/latest/gr/sigv4-calculate-signature.html

    Args:
        key (str): The AWS secret access key.
        date_stamp (str): The '%Y%m%d' date format.
        region_name (str): The AWS region.
        service_name (str): The AWS service name, eg. sts.

    Returns:
        str: The signing key bytes.
    """
    k_date = _sign(("AWS4" + key).encode("utf-8"), date_stamp)
    k_region = _sign(k_date, region_name)
    k_service = _sign(k_region, service_name)
    k_signing = _sign(k_service, "aws4_request")
    return k_signing


def _generate_authentication_header_map(
    host,
    canonical_uri,
    canonical_querystring,
    method,
    region,
    aws_security_credentials,
    request_payload="",
    additional_headers={},
):
    """Generates the authentication header map needed for generating the AWS
    Signature Version 4 signed request.

    Args:
        host (str): The AWS service URL hostname.
        canonical_uri (str): The AWS service URL path name.
        canonical_querystring (str): The AWS service URL query string.
        method (str): The HTTP method used to call this API.
        region (str): The AWS region.
        aws_security_credentials (AWSSecurityCredentials): The AWS security credentials.
        request_payload (Optional[str]): The optional request payload if
            available.
        additional_headers (Optional[Mapping[str, str]]): The optional
            additional headers needed for the requested AWS API.

    Returns:
        Mapping[str, str]: The AWS authentication header dictionary object.
            This contains the x-amz-date and authorization header information.
    """
    # iam.amazonaws.com host => iam service.
    # sts.us-east-2.amazonaws.com host => sts service.
    service_name = host.split(".")[0]

    current_time = _helpers.utcnow()
    amz_date = current_time.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = current_time.strftime("%Y%m%d")

    # Change all additional headers to be lower case.
    full_headers = {}
    for key in additional_headers:
        full_headers[key.lower()] = additional_headers[key]
    # Add AWS session token if available.
    if aws_security_credentials.session_token is not None:
        full_headers[
            _AWS_SECURITY_TOKEN_HEADER
        ] = aws_security_credentials.session_token

    # Required headers
    full_headers["host"] = host
    # Do not use generated x-amz-date if the date header is provided.
    # Previously the date was not fixed with x-amz- and could be provided
    # manually.
    # https://github.com/boto/botocore/blob/879f8440a4e9ace5d3cf145ce8b3d5e5ffb892ef/tests/unit/auth/aws4_testsuite/get-header-value-trim.req
    if "date" not in full_headers:
        full_headers[_AWS_DATE_HEADER] = amz_date

    # Header keys need to be sorted alphabetically.
    canonical_headers = ""
    header_keys = list(full_headers.keys())
    header_keys.sort()
    for key in header_keys:
        canonical_headers = "{}{}:{}\n".format(
            canonical_headers, key, full_headers[key]
        )
    signed_headers = ";".join(header_keys)

    payload_hash = hashlib.sha256((request_payload or "").encode("utf-8")).hexdigest()

    # https://docs.aws.amazon.com/general/latest/gr/sigv4-create-canonical-request.html
    canonical_request = "{}\n{}\n{}\n{}\n{}\n{}".format(
        method,
        canonical_uri,
        canonical_querystring,
        canonical_headers,
        signed_headers,
        payload_hash,
    )

    credential_scope = "{}/{}/{}/{}".format(
        date_stamp, region, service_name, _AWS_REQUEST_TYPE
    )

    # https://docs.aws.amazon.com/general/latest/gr/sigv4-create-string-to-sign.html
    string_to_sign = "{}\n{}\n{}\n{}".format(
        _AWS_ALGORITHM,
        amz_date,
        credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    )

    # https://docs.aws.amazon.com/general/latest/gr/sigv4-calculate-signature.html
    signing_key = _get_signing_key(
        aws_security_credentials.secret_access_key, date_stamp, region, service_name
    )
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    # https://docs.aws.amazon.com/general/latest/gr/sigv4-add-signature-to-request.html
    authorization_header = "{} Credential={}/{}, SignedHeaders={}, Signature={}".format(
        _AWS_ALGORITHM,
        aws_security_credentials.access_key_id,
        credential_scope,
        signed_headers,
        signature,
    )

    authentication_header = {"authorization_header": authorization_header}
    # Do not use generated x-amz-date if the date header is provided.
    if "date" not in full_headers:
        authentication_header["amz_date"] = amz_date
    return authentication_header


@dataclass
class AwsSecurityCredentials:
    """A class that models AWS security credentials with an optional session token.

        Attributes:
            access_key_id (str): The AWS security credentials access key id.
            secret_access_key (str): The AWS security credentials secret access key.
            session_token (Optional[str]): The optional AWS security credentials session token. This should be set when using temporary credentials.
    """

    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = None


class AwsSecurityCredentialsSupplier(metaclass=abc.ABCMeta):
    """Base class for AWS security credential suppliers. This can be implemented with custom logic to retrieve
    AWS security credentials to exchange for a Google Cloud access token. The AWS external account credential does
    not cache the AWS security credentials, so caching logic should be added in the implementation.
    """

    @abc.abstractmethod
    def get_aws_security_credentials(self, context, request):
        """Returns the AWS security credentials for the requested context.

        .. warning: This is not cached by the calling Google credential, so caching logic should be implemented in the supplier.

        Args:
            context (google.auth.externalaccount.SupplierContext): The context object
                containing information about the requested audience and subject token type.
            request (google.auth.transport.Request): The object used to make
                HTTP requests.

        Raises:
            google.auth.exceptions.RefreshError: If an error is encountered during
                security credential retrieval logic.

        Returns:
            AwsSecurityCredentials: The requested AWS security credentials.
        """
        raise NotImplementedError("")

    @abc.abstractmethod
    def get_aws_region(self, context, request):
        """Returns the AWS region for the requested context.

        Args:
            context (google.auth.externalaccount.SupplierContext): The context object
                containing information about the requested audience and subject token type.
            request (google.auth.transport.Request): The object used to make
                HTTP requests.

        Raises:
            google.auth.exceptions.RefreshError: If an error is encountered during
                region retrieval logic.

        Returns:
            str: The AWS region.
        """
        raise NotImplementedError("")


class _DefaultAwsSecurityCredentialsSupplier(AwsSecurityCredentialsSupplier):
    """Default implementation of AWS security credentials supplier. Supports retrieving
    credentials and region via EC2 metadata endpoints and environment variables.
    """

    def __init__(self, credential_source):
        self._region_url = credential_source.get("region_url")
        self._security_credentials_url = credential_source.get("url")
        self._imdsv2_session_token_url = credential_source.get(
            "imdsv2_session_token_url"
        )

    @_helpers.copy_docstring(AwsSecurityCredentialsSupplier)
    def get_aws_security_credentials(self, context, request):

        # Check environment variables for permanent credentials first.
        # https://docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html
        env_aws_access_key_id = os.environ.get(environment_vars.AWS_ACCESS_KEY_ID)
        env_aws_secret_access_key = os.environ.get(
            environment_vars.AWS_SECRET_ACCESS_KEY
        )
        # This is normally not available for permanent credentials.
        env_aws_session_token = os.environ.get(environment_vars.AWS_SESSION_TOKEN)
        if env_aws_access_key_id and env_aws_secret_access_key:
            return AwsSecurityCredentials(
                env_aws_access_key_id, env_aws_secret_access_key, env_aws_session_token
            )

        # Amazon ECS tasks, including tasks on the Fargate launch type, get
        # their task role credentials from the ECS container credential
        # provider rather than from EC2 instance metadata. Prefer it whenever
        # it is available, matching the AWS SDK credential chain; on Fargate it
        # is the only option, because the EC2 instance metadata server used
        # below is not routable from the task.
        container_credentials_url = self._get_container_credentials_url()
        if container_credentials_url is not None:
            credentials = self._get_container_security_credentials(
                request, container_credentials_url
            )
        else:
            imdsv2_session_token = self._get_imdsv2_session_token(request)
            role_name = self._get_metadata_role_name(request, imdsv2_session_token)

            # Get security credentials.
            credentials = self._get_metadata_security_credentials(
                request, role_name, imdsv2_session_token
            )

        return AwsSecurityCredentials(
            credentials.get("AccessKeyId"),
            credentials.get("SecretAccessKey"),
            credentials.get("Token"),
        )

    @_helpers.copy_docstring(AwsSecurityCredentialsSupplier)
    def get_aws_region(self, context, request):
        # The AWS metadata server is not available in some AWS environments
        # such as AWS lambda. Instead, it is available via environment
        # variable.
        env_aws_region = os.environ.get(environment_vars.AWS_REGION)
        if env_aws_region is not None:
            return env_aws_region

        env_aws_region = os.environ.get(environment_vars.AWS_DEFAULT_REGION)
        if env_aws_region is not None:
            return env_aws_region

        # On Amazon ECS the region is available from the task metadata
        # endpoint. This is the only in-task source on the Fargate launch type,
        # where the region URL below points at the EC2 instance metadata
        # server, which the task cannot reach.
        ecs_task_region = self._get_ecs_task_region(request)
        if ecs_task_region is not None:
            return ecs_task_region

        if not self._region_url:
            raise exceptions.RefreshError("Unable to determine AWS region")

        headers = None
        imdsv2_session_token = self._get_imdsv2_session_token(request)
        if imdsv2_session_token is not None:
            headers = {"X-aws-ec2-metadata-token": imdsv2_session_token}

        response = request(url=self._region_url, method="GET", headers=headers)

        # Support both string and bytes type response.data.
        response_body = (
            response.data.decode("utf-8")
            if hasattr(response.data, "decode")
            else response.data
        )

        if response.status != http_client.OK:
            raise exceptions.RefreshError(
                "Unable to retrieve AWS region: {}".format(response_body)
            )

        # This endpoint will return the region in format: us-east-2b.
        # Only the us-east-2 part should be used.
        return response_body[:-1]

    def _get_imdsv2_session_token(self, request):
        if request is not None and self._imdsv2_session_token_url is not None:
            headers = {
                "X-aws-ec2-metadata-token-ttl-seconds": _IMDSV2_SESSION_TOKEN_TTL_SECONDS
            }

            imdsv2_session_token_response = request(
                url=self._imdsv2_session_token_url, method="PUT", headers=headers
            )

            if imdsv2_session_token_response.status != http_client.OK:
                raise exceptions.RefreshError(
                    "Unable to retrieve AWS Session Token: {}".format(
                        imdsv2_session_token_response.data
                    )
                )

            return imdsv2_session_token_response.data
        else:
            return None

    def _get_metadata_security_credentials(
        self, request, role_name, imdsv2_session_token
    ):
        """Retrieves the AWS security credentials required for signing AWS
        requests from the AWS metadata server.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            role_name (str): The AWS role name required by the AWS metadata
                server security_credentials endpoint in order to return the
                credentials.
            imdsv2_session_token (str): The AWS IMDSv2 session token to be added as a
                header in the requests to AWS metadata endpoint.

        Returns:
            Mapping[str, str]: The AWS metadata server security credentials
                response.

        Raises:
            google.auth.exceptions.RefreshError: If an error occurs while
                retrieving the AWS security credentials.
        """
        headers = {"Content-Type": "application/json"}
        if imdsv2_session_token is not None:
            headers["X-aws-ec2-metadata-token"] = imdsv2_session_token

        response = request(
            url="{}/{}".format(self._security_credentials_url, role_name),
            method="GET",
            headers=headers,
        )

        # support both string and bytes type response.data
        response_body = (
            response.data.decode("utf-8")
            if hasattr(response.data, "decode")
            else response.data
        )

        if response.status != http_client.OK:
            raise exceptions.RefreshError(
                "Unable to retrieve AWS security credentials: {}".format(response_body)
            )

        credentials_response = json.loads(response_body)

        return credentials_response

    def _get_container_credentials_url(self):
        """Returns the ECS container credential provider URL for this task.

        The ECS agent injects the credential path into the container
        environment. Its presence is what makes the container credential
        provider usable; the ECS task metadata variables are injected on both
        the EC2 and Fargate launch types and say nothing about which credential
        endpoint is reachable, so they are not used for this decision.

        ``AWS_CONTAINER_CREDENTIALS_FULL_URI`` is deliberately not honored. It
        names an arbitrary host from the environment and is paired with a
        bearer token in ``AWS_CONTAINER_AUTHORIZATION_TOKEN``; ECS tasks are
        always served by the link-local provider below, so there is no reason
        to let a credential request, or that token, leave the host.

        Returns:
            Optional[str]: The ECS container credential provider URL, or None if
                the current environment does not expose one.

        Raises:
            google.auth.exceptions.RefreshError: If the credential path variable
                is set but empty.
        """
        relative_uri = os.environ.get(_ECS_CONTAINER_CREDENTIALS_RELATIVE_URI_ENV)
        if relative_uri is None:
            return None

        # Present but empty is a broken task environment, not the absence of
        # one. Falling through to the EC2 instance metadata server would sign
        # with the host's instance role on the EC2 launch type instead of the
        # task role, so fail closed. botocore's container provider likewise
        # keys off the variable's presence and never falls back.
        if not relative_uri:
            raise exceptions.RefreshError(
                "Unable to determine the ECS container credential provider "
                "endpoint: {} is empty".format(
                    _ECS_CONTAINER_CREDENTIALS_RELATIVE_URI_ENV
                )
            )

        # The path comes from the environment, so it is appended to the
        # link-local host rather than joined as a URL: that keeps the request
        # on the metadata endpoint even if the value looks like a full URL.
        return "{}/{}".format(_ECS_CONTAINER_CREDENTIALS_HOST, relative_uri.lstrip("/"))

    def _get_container_security_credentials(self, request, url):
        """Retrieves the AWS security credentials of the current ECS task role
        from the ECS container credential provider.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            url (str): The ECS container credential provider URL, as returned by
                :meth:`_get_container_credentials_url`.

        Returns:
            Mapping[str, str]: The ECS container credential provider response,
                in the same shape as the AWS metadata server response.

        Raises:
            google.auth.exceptions.RefreshError: If an error occurs while
                retrieving the AWS security credentials.
        """
        # The provider is reached over the link-local address and requires no
        # authorization header, so no token is sent to it.
        response = request(
            url=url, method="GET", headers={"Content-Type": "application/json"}
        )

        # The body is deliberately left out of both messages below. It reaches
        # logs, and a credentials document cut off in transit -- whether it
        # arrives as an error or fails to parse -- still carries a secret
        # access key. The status is checked before the body is decoded, so an
        # error response is never decoded at all.
        if response.status != http_client.OK:
            raise exceptions.RefreshError(
                "Unable to retrieve AWS security credentials from the ECS "
                "container credential provider: HTTP {}".format(response.status)
            )

        try:
            # support both string and bytes type response.data. Decoding is
            # inside the boundary: UnicodeDecodeError is a ValueError.
            response_body = (
                response.data.decode("utf-8")
                if hasattr(response.data, "decode")
                else response.data
            )
            credentials = json.loads(response_body)
        except ValueError:
            # Not raised from here, not even "from None": JSONDecodeError.doc
            # and UnicodeDecodeError.object hold the whole body, and an
            # exception raised inside this block would keep that reachable as
            # its __cause__ or __context__.
            credentials = None

        # Valid JSON that is not an object ([], null) is not a credentials
        # document either, and would otherwise fail later as an AttributeError.
        if not isinstance(credentials, dict):
            raise exceptions.RefreshError(
                "Unable to parse the AWS security credentials returned by the "
                "ECS container credential provider"
            )

        return credentials

    def _get_ecs_task_region(self, request):
        """Best effort lookup of the AWS region from the ECS task metadata
        endpoint.

        The ECS agent injects the task metadata base URI into every task
        container. The task response carries the task ARN, whose fourth field
        is the region.

        Returns None instead of raising when the region cannot be determined
        this way, so that the EC2 instance metadata server stays the fallback
        for workloads where it is reachable.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.

        Returns:
            Optional[str]: The AWS region, or None if it could not be determined.
        """
        metadata_uri = os.environ.get(
            _ECS_CONTAINER_METADATA_URI_V4_ENV
        ) or os.environ.get(_ECS_CONTAINER_METADATA_URI_ENV)
        if request is None or not metadata_uri:
            return None

        # The URI comes from the environment; only the link-local metadata
        # endpoint is ever contacted.
        if not _is_link_local(metadata_uri):
            return None

        response = request(url="{}/task".format(metadata_uri.rstrip("/")), method="GET")

        if response.status != http_client.OK:
            return None

        try:
            # support both string and bytes type response.data. Decoding is
            # inside the try: UnicodeDecodeError is a ValueError.
            response_body = (
                response.data.decode("utf-8")
                if hasattr(response.data, "decode")
                else response.data
            )
            task_arn = json.loads(response_body).get("TaskARN") or ""
        except (ValueError, AttributeError):
            # Not a JSON object, so not a task metadata response.
            return None

        if not isinstance(task_arn, str):
            return None

        # arn:<partition>:ecs:<region>:<account id>:task/...
        arn_fields = task_arn.split(":")
        if len(arn_fields) > 3 and arn_fields[3]:
            return arn_fields[3]

        return None

    def _get_metadata_role_name(self, request, imdsv2_session_token):
        """Retrieves the AWS role currently attached to the current AWS
        workload by querying the AWS metadata server. This is needed for the
        AWS metadata server security credentials endpoint in order to retrieve
        the AWS security credentials needed to sign requests to AWS APIs.

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
            imdsv2_session_token (str): The AWS IMDSv2 session token to be added as a
                header in the requests to AWS metadata endpoint.

        Returns:
            str: The AWS role name.

        Raises:
            google.auth.exceptions.RefreshError: If an error occurs while
                retrieving the AWS role name.
        """
        if self._security_credentials_url is None:
            raise exceptions.RefreshError(
                "Unable to determine the AWS metadata server security credentials endpoint"
            )

        headers = None
        if imdsv2_session_token is not None:
            headers = {"X-aws-ec2-metadata-token": imdsv2_session_token}

        response = request(
            url=self._security_credentials_url, method="GET", headers=headers
        )

        # support both string and bytes type response.data
        response_body = (
            response.data.decode("utf-8")
            if hasattr(response.data, "decode")
            else response.data
        )

        if response.status != http_client.OK:
            raise exceptions.RefreshError(
                "Unable to retrieve AWS role name {}".format(response_body)
            )

        return response_body


class Credentials(external_account.Credentials):
    """AWS external account credentials.
    This is used to exchange serialized AWS signature v4 signed requests to
    AWS STS GetCallerIdentity service for Google access tokens.
    """

    def __init__(
        self,
        audience,
        subject_token_type,
        token_url=external_account._DEFAULT_TOKEN_URL,
        credential_source=None,
        aws_security_credentials_supplier=None,
        *args,
        **kwargs
    ):
        """Instantiates an AWS workload external account credentials object.

        Args:
            audience (str): The STS audience field.
            subject_token_type (str): The subject token type based on the Oauth2.0 token exchange spec.
                Expected values include::

                    “urn:ietf:params:aws:token-type:aws4_request”

            token_url (Optional [str]): The STS endpoint URL. If not provided, will default to "https://sts.googleapis.com/v1/token".
            credential_source (Optional [Mapping]): The credential source dictionary used
                to provide instructions on how to retrieve external credential to be exchanged for Google access tokens.
                Either a credential source or an AWS security credentials supplier must be provided.

                Example credential_source for AWS credential::

                    {
                        "environment_id": "aws1",
                        "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15",
                        "region_url": "http://169.254.169.254/latest/meta-data/placement/availability-zone",
                        "url": "http://169.254.169.254/latest/meta-data/iam/security-credentials",
                        imdsv2_session_token_url": "http://169.254.169.254/latest/api/token"
                    }

            aws_security_credentials_supplier (Optional [AwsSecurityCredentialsSupplier]): Optional AWS security credentials supplier.
                This will be called to supply valid AWS security credentails which will then
                be exchanged for Google access tokens. Either an AWS security credentials supplier
                or a credential source must be provided.
            args (List): Optional positional arguments passed into the underlying :meth:`~external_account.Credentials.__init__` method.
            kwargs (Mapping): Optional keyword arguments passed into the underlying :meth:`~external_account.Credentials.__init__` method.

        Raises:
            google.auth.exceptions.RefreshError: If an error is encountered during
                access token retrieval logic.
            ValueError: For invalid parameters.

        .. note:: Typically one of the helper constructors
            :meth:`from_file` or
            :meth:`from_info` are used instead of calling the constructor directly.
        """
        super(Credentials, self).__init__(
            audience=audience,
            subject_token_type=subject_token_type,
            token_url=token_url,
            credential_source=credential_source,
            *args,
            **kwargs
        )
        if credential_source is None and aws_security_credentials_supplier is None:
            raise exceptions.InvalidValue(
                "A valid credential source or AWS security credentials supplier must be provided."
            )
        if (
            credential_source is not None
            and aws_security_credentials_supplier is not None
        ):
            raise exceptions.InvalidValue(
                "AWS credential cannot have both a credential source and an AWS security credentials supplier."
            )

        if aws_security_credentials_supplier:
            self._aws_security_credentials_supplier = aws_security_credentials_supplier
            # The regional cred verification URL would normally be provided through the credential source. So set it to the default one here.
            self._cred_verification_url = (
                _DEFAULT_AWS_REGIONAL_CREDENTIAL_VERIFICATION_URL
            )
        else:
            environment_id = credential_source.get("environment_id") or ""
            self._aws_security_credentials_supplier = _DefaultAwsSecurityCredentialsSupplier(
                credential_source
            )
            self._cred_verification_url = credential_source.get(
                "regional_cred_verification_url"
            )

            # Get the environment ID, i.e. "aws1". Currently, only one version supported (1).
            matches = re.match(r"^(aws)([\d]+)$", environment_id)
            if matches:
                env_id, env_version = matches.groups()
            else:
                env_id, env_version = (None, None)

            if env_id != "aws" or self._cred_verification_url is None:
                raise exceptions.InvalidResource(
                    "No valid AWS 'credential_source' provided"
                )
            elif env_version is None or int(env_version) != 1:
                raise exceptions.InvalidValue(
                    "aws version '{}' is not supported in the current build.".format(
                        env_version
                    )
                )

        self._target_resource = audience
        self._request_signer = None

    def retrieve_subject_token(self, request):
        """Retrieves the subject token using the credential_source object.
        The subject token is a serialized `AWS GetCallerIdentity signed request`_.

        The logic is summarized as:

        Retrieve the AWS region from the AWS_REGION or AWS_DEFAULT_REGION
        environment variable, from the Amazon ECS task metadata endpoint, or
        from the AWS metadata server availability-zone, in that order.

        Check AWS credentials in environment variables. If not found, retrieve
        them from the Amazon ECS container credential provider when the task
        environment exposes one, and otherwise from the AWS metadata server
        security-credentials endpoint.

        When retrieving AWS credentials from the metadata server
        security-credentials endpoint, the AWS role needs to be determined by
        calling the security-credentials endpoint without any argument. Then the
        credentials can be retrieved via: security-credentials/role_name

        Generate the signed request to AWS STS GetCallerIdentity action.

        Inject x-goog-cloud-target-resource into header and serialize the
        signed request. This will be the subject-token to pass to GCP STS.

        .. _AWS GetCallerIdentity signed request:
            https://cloud.google.com/iam/docs/access-resources-aws#exchange-token

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
        Returns:
            str: The retrieved subject token.
        """

        # Initialize the request signer if not yet initialized after determining
        # the current AWS region.
        if self._request_signer is None:
            self._region = self._aws_security_credentials_supplier.get_aws_region(
                self._supplier_context, request
            )
            self._request_signer = RequestSigner(self._region)

        # Retrieve the AWS security credentials needed to generate the signed
        # request.
        aws_security_credentials = self._aws_security_credentials_supplier.get_aws_security_credentials(
            self._supplier_context, request
        )
        # Generate the signed request to AWS STS GetCallerIdentity API.
        # Use the required regional endpoint. Otherwise, the request will fail.
        request_options = self._request_signer.get_request_options(
            aws_security_credentials,
            self._cred_verification_url.replace("{region}", self._region),
            "POST",
        )
        # The GCP STS endpoint expects the headers to be formatted as:
        # [
        #   {key: 'x-amz-date', value: '...'},
        #   {key: 'Authorization', value: '...'},
        #   ...
        # ]
        # And then serialized as:
        # quote(json.dumps({
        #   url: '...',
        #   method: 'POST',
        #   headers: [{key: 'x-amz-date', value: '...'}, ...]
        # }))
        request_headers = request_options.get("headers")
        # The full, canonical resource name of the workload identity pool
        # provider, with or without the HTTPS prefix.
        # Including this header as part of the signature is recommended to
        # ensure data integrity.
        request_headers["x-goog-cloud-target-resource"] = self._target_resource

        # Serialize AWS signed request.
        aws_signed_req = {}
        aws_signed_req["url"] = request_options.get("url")
        aws_signed_req["method"] = request_options.get("method")
        aws_signed_req["headers"] = []
        # Reformat header to GCP STS expected format.
        for key in request_headers.keys():
            aws_signed_req["headers"].append(
                {"key": key, "value": request_headers[key]}
            )

        return urllib.parse.quote(
            json.dumps(aws_signed_req, separators=(",", ":"), sort_keys=True)
        )

    def _create_default_metrics_options(self):
        metrics_options = super(Credentials, self)._create_default_metrics_options()
        metrics_options["source"] = "aws"
        if self._has_custom_supplier():
            metrics_options["source"] = "programmatic"
        return metrics_options

    def _has_custom_supplier(self):
        return self._credential_source is None

    def _constructor_args(self):
        args = super(Credentials, self)._constructor_args()
        # If a custom supplier was used, append it to the args dict.
        if self._has_custom_supplier():
            args.update(
                {
                    "aws_security_credentials_supplier": self._aws_security_credentials_supplier
                }
            )
        return args

    @classmethod
    def from_info(cls, info, **kwargs):
        """Creates an AWS Credentials instance from parsed external account info.

        Args:
            info (Mapping[str, str]): The AWS external account info in Google
                format.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.aws.Credentials: The constructed credentials.

        Raises:
            ValueError: For invalid parameters.
        """
        aws_security_credentials_supplier = info.get(
            "aws_security_credentials_supplier"
        )
        kwargs.update(
            {"aws_security_credentials_supplier": aws_security_credentials_supplier}
        )
        return super(Credentials, cls).from_info(info, **kwargs)

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Creates an AWS Credentials instance from an external account json file.

        Args:
            filename (str): The path to the AWS external account json file.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            google.auth.aws.Credentials: The constructed credentials.
        """
        return super(Credentials, cls).from_file(filename, **kwargs)

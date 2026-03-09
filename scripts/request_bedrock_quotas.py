"""
Request AWS Bedrock quota increases for classroom use.

Usage:
    # Step 1: List current quotas to see codes and values
    python scripts/request_bedrock_quotas.py --list

    # Step 2: Request increases for all hackathon models
    python scripts/request_bedrock_quotas.py --request

Uses the AWS_PROFILE environment variable or --profile flag.
Example: AWS_PROFILE=de2 python scripts/request_bedrock_quotas.py --list
"""

import argparse
import boto3
import sys

# Models from HACKATHON.md — keywords to match quota names (lowercased)
HACKATHON_MODELS = [
    "nova lite",
    "nova micro",
    "mistral 7b",
    "claude 3 haiku",
]

# Desired quotas for 20 students (5 teams of 4) with headroom
# Tool-calling agents make 3-5 API calls per interaction
DESIRED_RPM = 300       # requests per minute
DESIRED_TPM = 600_000   # tokens per minute


def get_client(profile: str, region: str):
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("service-quotas")


def list_bedrock_quotas(client, filter_models=True):
    """List all Bedrock quotas, optionally filtered to hackathon models."""
    quotas = []
    paginator = client.get_paginator("list_service_quotas")

    for page in paginator.paginate(ServiceCode="bedrock"):
        for q in page["Quotas"]:
            quotas.append(q)

    if filter_models:
        keywords = HACKATHON_MODELS
        quotas = [
            q for q in quotas
            if any(kw in q["QuotaName"].lower() for kw in keywords)
        ]

    # Sort by name for readability
    quotas.sort(key=lambda q: q["QuotaName"])
    return quotas


def print_quotas(quotas):
    print(f"\n{'Quota Name':<80} {'Code':<15} {'Value':>10} {'Adjustable'}")
    print("-" * 120)
    for q in quotas:
        adjustable = "Yes" if q.get("Adjustable", False) else "No"
        print(f"{q['QuotaName']:<80} {q['QuotaCode']:<15} {q['Value']:>10.0f} {adjustable}")
    print(f"\nTotal: {len(quotas)} quotas found")


def request_increases(client, quotas, desired_rpm, desired_tpm):
    """Request quota increases for quotas that are below desired values."""
    results = []

    for q in quotas:
        if not q.get("Adjustable", False):
            print(f"  SKIP (not adjustable): {q['QuotaName']}")
            continue

        name_lower = q["QuotaName"].lower()

        # Determine desired value based on quota type
        if "tokens" in name_lower:
            desired = desired_tpm
        elif "request" in name_lower:
            desired = desired_rpm
        else:
            print(f"  SKIP (unknown type): {q['QuotaName']}")
            continue

        if q["Value"] >= desired:
            print(f"  OK (already {q['Value']:.0f} >= {desired}): {q['QuotaName']}")
            continue

        print(f"  REQUESTING {q['Value']:.0f} -> {desired}: {q['QuotaName']}")
        try:
            response = client.request_service_quota_increase(
                ServiceCode="bedrock",
                QuotaCode=q["QuotaCode"],
                DesiredValue=float(desired),
            )
            status = response["RequestedQuota"]["Status"]
            req_id = response["RequestedQuota"]["Id"]
            results.append((q["QuotaName"], status, req_id))
            print(f"    -> Status: {status}, Request ID: {req_id}")
        except client.exceptions.ResourceAlreadyExistsException:
            print(f"    -> Already has a pending request")
            results.append((q["QuotaName"], "ALREADY_PENDING", ""))
        except Exception as e:
            print(f"    -> ERROR: {e}")
            results.append((q["QuotaName"], "ERROR", str(e)))

    return results


def main():
    parser = argparse.ArgumentParser(description="Manage Bedrock quotas for classroom use")
    parser.add_argument("--profile", default="de2", help="AWS profile name (default: de2)")
    parser.add_argument("--region", default="eu-west-1", help="AWS region (default: eu-west-1)")
    parser.add_argument("--list", action="store_true", help="List current Bedrock quotas")
    parser.add_argument("--list-all", action="store_true", help="List ALL Bedrock quotas (not just hackathon models)")
    parser.add_argument("--request", action="store_true", help="Request quota increases")
    parser.add_argument("--rpm", type=int, default=DESIRED_RPM, help=f"Desired requests/min (default: {DESIRED_RPM})")
    parser.add_argument("--tpm", type=int, default=DESIRED_TPM, help=f"Desired tokens/min (default: {DESIRED_TPM})")
    args = parser.parse_args()

    if not args.list and not args.list_all and not args.request:
        parser.print_help()
        sys.exit(1)

    print(f"Using profile: {args.profile}, region: {args.region}")
    client = get_client(args.profile, args.region)

    if args.list or args.list_all:
        print("Fetching Bedrock quotas...")
        quotas = list_bedrock_quotas(client, filter_models=not args.list_all)
        print_quotas(quotas)

    if args.request:
        print(f"\nFetching quotas for hackathon models...")
        quotas = list_bedrock_quotas(client, filter_models=True)
        print(f"Found {len(quotas)} relevant quotas\n")
        print(f"Requesting: {args.rpm} RPM, {args.tpm} TPM\n")
        results = request_increases(client, quotas, args.rpm, args.tpm)

        print(f"\n{'='*60}")
        print(f"Summary: {len(results)} requests made")
        for name, status, req_id in results:
            print(f"  [{status}] {name}")


if __name__ == "__main__":
    main()

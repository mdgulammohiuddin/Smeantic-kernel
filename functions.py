import json
import re
import os
from azure.identity import ClientSecretCredential
from azure.mgmt.resource import ResourceManagementClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
AZURE_SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")


def get_resource_name_by_partial_match(resource_name: str, resource_type: str = None) -> str:
    def normalize(text):
        return re.sub(r'[\s\-_]', '', text.lower())

    try:
        credential = ClientSecretCredential(
            tenant_id=AZURE_TENANT_ID,
            client_id=AZURE_CLIENT_ID,
            client_secret=AZURE_CLIENT_SECRET
        )
        resource_client = ResourceManagementClient(credential, AZURE_SUBSCRIPTION_ID)

        normalized_input = normalize(resource_name)
        exact_match = None
        loose_match = None

        for resource in resource_client.resources.list():
            if resource_type and resource_type.lower() not in resource.type.lower():
                continue

            normalized_resource_name = normalize(resource.name)

            if normalized_resource_name == normalized_input:
                exact_match = resource
                break
            elif normalized_input in normalized_resource_name and loose_match is None:
                loose_match = resource

        final_match = exact_match or loose_match

        if final_match:
            return json.dumps({
                "status": "success",
                "resource_name": final_match.name,
                "resource_id": final_match.id,
                "resource_type": final_match.type,
                "resource_group": final_match.id.split('/')[4]
            })
        else:
            return json.dumps({
                "status": "not_found",
                "message": f"No resource found matching '{resource_name}'"
            })

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Error: {str(e)}"
        })

from azure.storage.blob import BlobServiceClient
import os

def upload_to_azure_blob(local_file_path, container_name, blob_name, connection_string):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        with open(local_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        print(f"Uploaded to Azure Blob: {blob_name}")
    except Exception as e:
        print(f"Azure upload failed: {e}")

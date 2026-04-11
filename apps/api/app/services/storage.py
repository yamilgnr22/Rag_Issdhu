import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings


@dataclass
class StoredObject:
    uri: str
    size: int


class LocalObjectStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_prefix = self.root.as_posix()

    def save_bytes(self, relative_path: str, content: bytes) -> StoredObject:
        target = self.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return StoredObject(uri=target.as_posix(), size=len(content))

    def save_json(self, relative_path: str, payload: object) -> StoredObject:
        target = self.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8")
        target.write_bytes(raw)
        return StoredObject(uri=target.as_posix(), size=len(raw))

    def delete_prefix(self, relative_prefix: str) -> None:
        target = self.root / relative_prefix
        if not target.exists():
            return
        if target.is_file():
            target.unlink(missing_ok=True)
            return
        shutil.rmtree(target, ignore_errors=True)


class MinioObjectStore:
    def __init__(self, settings: Settings) -> None:
        import boto3

        self.bucket = settings.minio_bucket
        self.base_prefix = f"s3://{self.bucket}"
        endpoint_url = f"http{'s' if settings.minio_secure else ''}://{settings.minio_endpoint}"
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            region_name="us-east-1",
        )
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        buckets = self.client.list_buckets().get("Buckets", [])
        if not any(bucket["Name"] == self.bucket for bucket in buckets):
            self.client.create_bucket(Bucket=self.bucket)

    def save_bytes(self, relative_path: str, content: bytes) -> StoredObject:
        self.client.put_object(Bucket=self.bucket, Key=relative_path, Body=content)
        return StoredObject(uri=f"s3://{self.bucket}/{relative_path}", size=len(content))

    def save_json(self, relative_path: str, payload: object) -> StoredObject:
        raw = json.dumps(payload, ensure_ascii=True, indent=2).encode("utf-8")
        self.client.put_object(
            Bucket=self.bucket,
            Key=relative_path,
            Body=raw,
            ContentType="application/json",
        )
        return StoredObject(uri=f"s3://{self.bucket}/{relative_path}", size=len(raw))

    def delete_prefix(self, relative_prefix: str) -> None:
        prefix = str(relative_prefix or "").strip().strip("/")
        if not prefix:
            return
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            contents = page.get("Contents", [])
            if not contents:
                continue
            delete_payload = {
                "Objects": [{"Key": item["Key"]} for item in contents if item.get("Key")],
                "Quiet": True,
            }
            if delete_payload["Objects"]:
                self.client.delete_objects(Bucket=self.bucket, Delete=delete_payload)


def build_object_store(settings: Settings) -> LocalObjectStore | MinioObjectStore:
    if settings.storage_backend == "minio":
        return MinioObjectStore(settings)
    return LocalObjectStore(settings.local_storage_root)

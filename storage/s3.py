from pathlib import Path

from config.settings import config


# ──────────────────────────────────────────────
#  Local backend
# ──────────────────────────────────────────────

def _local_path(s3_key: str) -> Path:
    path = config.local_upload_dir / s3_key
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _local_upload(file_bytes: bytes, s3_key: str, **_) -> str:
    _local_path(s3_key).write_bytes(file_bytes)
    return s3_key


def _local_download(s3_key: str) -> bytes:
    return _local_path(s3_key).read_bytes()


def _local_delete(s3_key: str) -> None:
    path = _local_path(s3_key)
    breakpoint()
    if path.exists():
        path.unlink()


# ──────────────────────────────────────────────
#  S3 backend
# ──────────────────────────────────────────────

def _s3_client():
    import boto3
    return boto3.client("s3", region_name=config.aws_region)


def _s3_upload(file_bytes: bytes, s3_key: str, content_type: str = "application/octet-stream") -> str:
    _s3_client().put_object(
        Bucket=config.s3_bucket,
        Key=s3_key,
        Body=file_bytes,
        ContentType=content_type,
    )
    return s3_key


def _s3_download(s3_key: str) -> bytes:
    response = _s3_client().get_object(Bucket=config.s3_bucket, Key=s3_key)
    return response["Body"].read()


def _s3_delete(s3_key: str) -> None:
    try:
        _s3_client().delete_object(Bucket=config.s3_bucket, Key=s3_key)
    except Exception:
        pass


# ──────────────────────────────────────────────
#  Public API — routes to the configured backend
# ──────────────────────────────────────────────

def upload_file(file_bytes: bytes, s3_key: str, content_type: str = "application/octet-stream") -> str:
    if config.storage_backend == "s3":
        return _s3_upload(file_bytes, s3_key, content_type)
    return _local_upload(file_bytes, s3_key)


def download_file(s3_key: str) -> bytes:
    if config.storage_backend == "s3":
        return _s3_download(s3_key)
    return _local_download(s3_key)


def delete_file(s3_key: str) -> None:
    if config.storage_backend == "s3":
        _s3_delete(s3_key)
    else:
        _local_delete(s3_key)

"""Contracts for exact Gemini subscription routes on the account gateway."""
import re
import base64
import copy
import io
import json
from urllib.parse import urlsplit

from PIL import Image


REQUEST_BUDGET_BYTES = 1_000_000  # Headroom below the gateway's 1 MiB limit.


def fit_image_request(request):
    """Bound inline image requests without dropping frames, text, or resolution.

    Only oversized requests are recompressed. The caller records the returned
    transport metadata; original benchmark image artifacts remain untouched.
    Fail locally if even quality 50 cannot fit, rather than silently resizing.
    """
    request = copy.deepcopy(request)

    def body_size():
        body = {k: v for k, v in request.items() if k not in {'timeout', 'extra_body'}}
        body.update(request.get('extra_body', {}))
        return len(json.dumps(body, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8'))

    original_size = body_size()
    metadata = {'original_body_bytes': original_size, 'body_bytes': original_size,
                'budget_bytes': REQUEST_BUDGET_BYTES, 'recompressed': False}
    if original_size <= REQUEST_BUDGET_BYTES:
        return request, metadata
    images = []
    for message in request['messages']:
        if not isinstance(message.get('content'), list):
            continue
        for part in message['content']:
            if part.get('type') != 'image_url':
                continue
            target = part['image_url']
            url = target['url']
            if not url.startswith('data:image/') or ';base64,' not in url:
                raise ValueError('Oversized account-pool request requires inline images')
            with Image.open(io.BytesIO(base64.b64decode(url.split(';base64,', 1)[1], validate=True))) as image:
                if image.mode not in {'RGB', 'L'}:
                    raise ValueError('Account-pool compression requires RGB or grayscale images')
                images.append((target, image.copy()))
    for quality in (95, 90, 85, 80, 75, 65, 50) if images else ():
        for target, image in images:
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality, subsampling=0)
            target['url'] = 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
        size = body_size()
        if size <= REQUEST_BUDGET_BYTES:
            metadata.update(body_bytes=size, recompressed=True, image_encoding='jpeg',
                            jpeg_quality=quality, jpeg_subsampling=0,
                            image_count=len(images), image_sizes=[list(im.size) for _, im in images])
            return request, metadata
    raise ValueError('Account-pool request exceeds the image payload budget; no frames were dropped or resized')


def validate_route(model, effort, base_url):
    url = urlsplit(base_url or "")
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment or url.path.rstrip("/") != "/v1":
        raise ValueError("Account pool requires an explicit http(s)://host[:port]/v1 base URL")
    pinned = {
        "antigravity/gemini-4-argon-medium": "medium",
        "antigravity/claude-sonnet-5-5-medium": "medium",
        "codex/gpt-6-astra": "medium",
    }
    if model not in pinned or effort != pinned[model]:
        raise ValueError("This pilot requires an explicitly pinned model route and medium effort")

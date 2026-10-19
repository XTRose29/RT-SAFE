"""Check the real SDK wire body against the gateway image limit, offline."""
import base64
import copy
import io
import json
import random
import unittest

try:
    import httpx2 as httpx
except ImportError:
    import httpx
import openai
from PIL import Image

from llm.account_pool import fit_image_request, REQUEST_BUDGET_BYTES


class PayloadTests(unittest.TestCase):
    def request(self):
        rng = random.Random(42)
        content = [{'type': 'text', 'text': 'Keep all frames in order. 世界'}]
        for size in ((480, 360), (480, 361), (480, 362)):
            image = Image.frombytes('RGB', size, rng.randbytes(size[0] * size[1] * 3))
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            content.append({'type': 'image_url', 'image_url': {
                'url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode(), 'detail': 'high'}})
        return dict(model='antigravity/gemini-4-argon-low',
                    messages=[{'role': 'user', 'content': content}],
                    timeout=10, extra_body={'allow_paid': False})

    def test_oversized_request_fits_actual_sdk_wire_without_losing_frames(self):
        original = self.request()
        saved = copy.deepcopy(original)
        request, meta = fit_image_request(original)
        self.assertEqual(original, saved)
        self.assertTrue(meta['recompressed'])
        self.assertGreater(meta['original_body_bytes'], 1048576)
        self.assertLessEqual(meta['body_bytes'], REQUEST_BUDGET_BYTES)
        content = request['messages'][0]['content']
        self.assertEqual(content[0], original['messages'][0]['content'][0])
        self.assertEqual(len(content), 4)
        for part, size in zip(content[1:], ((480, 360), (480, 361), (480, 362))):
            self.assertEqual(part['image_url']['detail'], 'high')
            raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
            with Image.open(io.BytesIO(raw)) as image:
                self.assertEqual(image.format, 'JPEG')
                self.assertEqual(image.size, size)

        def handle(wire):
            self.assertEqual(len(wire.content), meta['body_bytes'])
            self.assertFalse(json.loads(wire.content)['allow_paid'])
            return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion',
                'created': 0, 'model': request['model'], 'choices': []})
        with openai.OpenAI(api_key='offline-test', http_client=httpx.Client(
                transport=httpx.MockTransport(handle))) as client:
            client.chat.completions.create(**request)

    def test_small_request_is_unchanged(self):
        request = dict(model='test', messages=[{'role': 'user', 'content': 'hello'}])
        fitted, meta = fit_image_request(request)
        self.assertEqual(fitted, request)
        self.assertFalse(meta['recompressed'])

    def test_impossible_budget_fails_without_truncating_text(self):
        request = self.request()
        request['messages'][0]['content'][0]['text'] = 'x' * REQUEST_BUDGET_BYTES
        with self.assertRaisesRegex(ValueError, 'no frames were dropped or resized'):
            fit_image_request(request)


if __name__ == '__main__':
    unittest.main()

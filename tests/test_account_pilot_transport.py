"""Pin the pilot routes, effort, no-paid-fallback policy, and served identity."""
import json
import openai
import pytest
from PIL import Image
try:
    import httpx2 as httpx
except ImportError:
    import httpx
from llm.rt_llm import RTLLM
from llm.account_pool import validate_route

MODELS=['antigravity/gemini-4-argon-medium','codex/gpt-6-astra','antigravity/claude-sonnet-5-5-medium']

@pytest.mark.parametrize('model', MODELS)
def test_pilot_wire_pins_effort_disables_paid_and_preserves_image(monkeypatch,model):
    monkeypatch.setenv('POOL_KEY','offline-only')
    def handle(request):
        body=json.loads(request.content)
        assert body['model']==model
        assert body['allow_paid'] is False
        assert 'temperature' not in body
        assert body['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/')
        if model.startswith('codex/'):
            assert body['reasoning_effort']=='medium'
        else:
            assert 'reasoning_effort' not in body
        return httpx.Response(200,json={'id':'offline','object':'chat.completion','created':0,
            'model':model,'choices':[{'index':0,'message':{'role':'assistant','content':'Action: wait\nParam: 1'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}})
    llm=RTLLM(model,url='http://example.test/v1',provider='account-pool',reasoning=True,
              reasoning_effort='medium',extra_body={'allow_paid':True},raise_on_api_error=True)
    with openai.OpenAI(api_key='offline-only',base_url='http://example.test/v1',http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        llm.client=client
        result=llm.generate_response_openai('system','user',[Image.new('RGB',(5,5))])
    assert result[0].action_param=='1'
    assert llm.last_usage_metadata['served_model']==model
    assert llm.last_usage_metadata['account_pool']['allow_paid'] is False

@pytest.mark.parametrize('model,effort',[('claude/sonnet','medium'),('codex/gpt-6-astra','high'),('antigravity/gemini-4-argon-low','medium')])
def test_unpinned_routes_or_mismatched_effort_rejected(model,effort):
    with pytest.raises(ValueError):validate_route(model,effort,'http://example.test/v1')

def test_gateway_substitution_is_rejected(monkeypatch):
    monkeypatch.setenv('POOL_KEY','offline-only')
    def handle(request):
        return httpx.Response(200,json={'id':'offline','object':'chat.completion','created':0,
            'model':'different-model','choices':[{'index':0,'message':{'role':'assistant','content':'Action: wait\nParam: 1'},'finish_reason':'stop'}]})
    llm=RTLLM(MODELS[0],url='http://example.test/v1',provider='account-pool',reasoning=True,
              reasoning_effort='medium',raise_on_api_error=True)
    with openai.OpenAI(api_key='offline-only',base_url='http://example.test/v1',http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        llm.client=client
        with pytest.raises(ValueError,match='different model'):
            llm.generate_response_openai('system','user',[])

"""验证主题看板的入口、资源、数据透传与图表语义。"""

import gzip
import json
import shutil
import subprocess
from pathlib import Path
from wsgiref.util import setup_testing_defaults

import pytest

from hscredit import cli
from hscredit.dashboard import _ThemedDashboard, _patch_plotly_bundle, _theme_config


def _request(app, path, method="GET", **extra):
    environ = {}
    setup_testing_defaults(environ)
    environ.update(PATH_INFO=path, REQUEST_METHOD=method, **extra)
    result = {}
    iterator = app(environ, lambda status, headers, exc_info=None: result.update(status=status, headers=dict(headers)))
    result["body"] = b"".join(iterator)
    if hasattr(iterator, "close"):
        iterator.close()
    return result


def _upstream(environ, start_response):
    path = environ["PATH_INFO"]
    if path == "/static/bundle.js":
        payload = b"var core={};out._doPlot=core._doPlot,out.newPlot=core.newPlot;"
        headers = [("Content-Type", "application/javascript")]
    elif path.startswith("/dashboard"):
        payload = b'<html><head><script defer src="/static/bundle.js"></script></head><body>Optuna</body></html>'
        payload = gzip.compress(payload)
        headers = [("Content-Type", "text/html"), ("Content-Encoding", "gzip"), ("Content-Security-Policy", "script-src 'self'")]
    else:
        payload = b'{"values":[0.7,0.2],"state":"COMPLETE"}'
        headers = [("Content-Type", "application/json")]
    start_response("200 OK", headers + [("Content-Length", str(len(payload)))])
    return [payload]


def test_page_loads_theme_before_bundle_and_keeps_csp():
    app = _ThemedDashboard(_upstream)
    response = _request(app, "/dashboard/studies/1", HTTP_IF_NONE_MATCH='"old"')
    assert response["status"] == "200 OK"
    assert response["body"].index(b"/_hscredit_dashboard/theme.js") < response["body"].index(b"/static/bundle.js")
    assert response["headers"]["Content-Security-Policy"] == "script-src 'self'"
    assert "Content-Encoding" not in response["headers"]
    assert int(response["headers"]["Content-Length"]) == len(response["body"])
    head = _request(app, "/dashboard/studies/1", method="HEAD")
    assert head["body"] == b""
    assert head["headers"]["Content-Length"] == response["headers"]["Content-Length"]


def test_api_and_mutations_are_forwarded_without_touching_data():
    app = _ThemedDashboard(_upstream)
    assert _request(app, "/api/studies/1") == _request(_upstream, "/api/studies/1")
    assert _request(app, "/api/studies/1/note", method="POST") == _request(_upstream, "/api/studies/1/note", method="POST")


def test_font_cache_and_asset_paths_are_bounded():
    app = _ThemedDashboard(_upstream)
    font = _request(app, "/_hscredit_dashboard/font.ttf")
    assert font["headers"]["Content-Type"] == "font/ttf"
    assert len(font["body"]) > 1000
    cached = _request(app, "/_hscredit_dashboard/font.ttf", HTTP_IF_NONE_MATCH=font["headers"]["ETag"])
    assert cached["status"] == "304 Not Modified" and cached["body"] == b""
    assert _request(app, "/_hscredit_dashboard/../fonts/font.ttf")["status"] == "404 Not Found"


def test_unrecognized_frontend_fails_explicitly():
    with pytest.raises(RuntimeError, match="前端结构不受"):
        _patch_plotly_bundle(b"upstream frontend changed")


def test_cli_passes_storage_and_local_server_options(monkeypatch):
    calls = []
    monkeypatch.setattr("hscredit.dashboard.run_dashboard", lambda storage, **kwargs: calls.append((storage, kwargs)))
    assert cli.main(["dashboard", "sqlite:///demo.db", "--port", "8088", "--quiet"]) == 0
    assert calls == [("sqlite:///demo.db", {"host": "127.0.0.1", "port": 8088, "artifact_dir": None, "quiet": True})]
    with pytest.raises(SystemExit):
        cli.main(["dashboard", "sqlite:///demo.db", "--port", "0"])


def test_installed_optuna_frontend_and_study_api():
    optuna = pytest.importorskip("optuna")
    pytest.importorskip("optuna_dashboard")
    from hscredit.dashboard import create_dashboard_app

    storage = optuna.storages.InMemoryStorage()
    study = optuna.create_study(storage=storage, study_name="主题验证")
    study.add_trial(optuna.trial.create_trial(value=0.73))
    app = create_dashboard_app(storage)
    response = _request(app, "/api/studies")
    assert response["status"] == "200 OK"
    assert json.loads(response["body"])["study_summaries"][0]["study_name"] == "主题验证"
    assert b"__hscreditDashboard.wrapPlotly" in _request(app, "/static/bundle.js")["body"]


@pytest.mark.skipif(shutil.which("node") is None, reason="图表主题脚本校验需要 Node.js")
def test_plotly_theme_preserves_values_and_reapplies_on_react():
    theme = Path(__file__).resolve().parents[2] / "hscredit/resources/dashboard/theme.js"
    javascript = (
        "const config="
        + json.dumps(_theme_config())
        + ";const source="
        + json.dumps(str(theme))
        + ";"
        + r"""
const assert = require('node:assert/strict');
global.window = {__hscreditDashboardConfig: config};
global.document = {querySelector:()=>null, documentElement:{dataset:{}}, addEventListener:()=>{}};
eval(require('node:fs').readFileSync(source,'utf8'));
const input=[{type:'scatter',x:[1,2],y:[.7,.8],marker:{color:[0,1],colorbar:{title:{text:'Trial'}}}},
 {type:'scatter',x:[2],y:[.8],marker:{color:[1],colorbar:{title:{text:'Best Trial'}}}},
 {name:'AUC',x:[1,2],y:[.7,.8],line:{color:'green'},mode:'lines'}];
const layout={xaxis:{range:[0,3]},title:{text:'模型效果'},uirevision:'stable'};
const before=JSON.stringify([input,layout]);
let calls=[];
const api={newPlot:(...args)=>{calls.push(args);return 'new';},react:(...args)=>{calls.push(args);return 'update';}};
window.__hscreditDashboard.wrapPlotly(api);
window.__hscreditDashboard.wrapPlotly(api);
assert.equal(api.newPlot('graph',input,layout,{responsive:true}),'new');
assert.equal(api.react('graph',input,layout,{responsive:true}),'update');
for(const args of calls){
 assert.deepEqual(args[1][0].x,[1,2]);assert.deepEqual(args[1][0].y,[.7,.8]);
 assert.deepEqual(args[1][0].marker.color,[0,1]);
 assert.equal(args[1][0].marker.colorscale.at(-1)[1],config.primary);
 assert.equal(args[1][1].marker.colorscale.at(-1)[1],config.secondary);
 assert.equal(args[1][2].line.color,config.metrics.auc);
 assert.deepEqual(args[2].xaxis.range,[0,3]);assert.equal(args[2].uirevision,'stable');
 assert.ok(args[2].font.family.includes(config.font));assert.equal(args[3].responsive,true);
}
assert.equal(JSON.stringify([input,layout]),before);
const format=window.__hscreditDashboard.formatAttributeText;
const attributes='{"\\u6307\\u6807\\u540d\\u79f0":["AUC","KS_DIFF"],"嵌套":{"备注":"中文","空值":null,"启用":true},"大整数":9007199254740993,"精度":0.1234567890123456789}';
const readable=format(attributes);
assert.ok(readable.includes('"指标名称": ['));
assert.ok(readable.includes('\n    "备注": "中文"'));
assert.ok(readable.includes('9007199254740993'));
assert.ok(readable.includes('0.1234567890123456789'));
assert.equal(format(readable),readable);
assert.deepEqual(JSON.parse(readable),JSON.parse(attributes));
assert.equal(format('普通中文 / C:\\users\\test'),null);
assert.equal(format('{"被截断":"\\u6307'),null);
assert.equal(format('[]'),'[]');
assert.equal(format('{}'),'{}');
const literal={pattern:String.raw`\u6307`,html:'<img src=x onerror=alert(1)>'};
assert.deepEqual(JSON.parse(format(JSON.stringify(literal))),literal);
"""
    )
    subprocess.run([shutil.which("node"), "-e", javascript], check=True, capture_output=True, text=True, encoding="utf-8")

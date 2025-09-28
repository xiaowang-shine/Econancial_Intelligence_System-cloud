import json
import threading
from pathlib import Path
import sys

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import app, task_manager

TEST_DIR = Path(__file__).parent
MONTHLY_FILE = TEST_DIR / "test_files" / "test_monthly.xlsx"


@pytest.fixture(autouse=True)
def clean_task_manager():
    task_manager.tasks.clear()
    yield
    task_manager.tasks.clear()


@pytest.fixture
def client(tmp_path, monkeypatch):
    app.config['TESTING'] = True
    # 使用临时目录存储上传文件，避免污染真实目录
    app.config['UPLOAD_FOLDER'] = tmp_path
    monkeypatch.setenv('DOWNLOAD_DIR', str(tmp_path / "downloads"))
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def stub_training(monkeypatch):
    fake_result = {
        'forecast': [{
            'date': '2025-01-31',
            'value': 123.0,
            'lower': 110.0,
            'upper': 135.0
        }],
        'kpi': {'debt_ratio': 40},
        'explain': {'feature_importance': []},
        'suggestions': [],
        'meta': {'model': 'TestModel', 'fluctuation': 0.0}
    }

    def fake_run_training_task(monthly_df, health_df, mapping):
        return fake_result

    monkeypatch.setattr('model.routes.run_training_task', fake_run_training_task)
    return fake_result


@pytest.fixture
def immediate_thread(monkeypatch):
    class ImmediateThread:
        def __init__(self, target, args=(), kwargs=None, daemon=True):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            self._target(*self._args, **self._kwargs)

    monkeypatch.setattr(threading, 'Thread', lambda target, args=(), kwargs=None, daemon=True: ImmediateThread(target, args, kwargs, daemon))


def test_upload_preview_only_monthly(client):
    with MONTHLY_FILE.open('rb') as monthly:
        # Flask test_client 使用 data 参数上传文件，格式为 (file_object, filename, content_type)
        data = {'fileMonthly': (monthly, MONTHLY_FILE.name, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
        response = client.post('/upload_preview', data=data, content_type='multipart/form-data')

    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.get_data(as_text=True)}"
    payload = response.get_json()
    assert payload['status'] == 'ok'
    assert 'token' in payload
    assert 'columns' in payload and 'fileMonthly' in payload['columns']
    assert 'preview' in payload and 'fileMonthly' in payload['preview']


def test_task_flow(client, stub_training, immediate_thread):
    with MONTHLY_FILE.open('rb') as monthly:
        # Flask test_client 使用 data 参数上传文件，格式为 (file_object, filename, content_type)
        data = {'fileMonthly': (monthly, MONTHLY_FILE.name, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
        response = client.post('/start_task', data=data, content_type='multipart/form-data')

    assert response.status_code == 200
    payload = response.get_json()
    assert payload['status'] == 'ok'
    task_id = payload['task_id']
    assert task_id

    # 查询任务状态
    status_resp = client.get(f'/task_status?task_id={task_id}')
    assert status_resp.status_code == 200
    status_payload = status_resp.get_json()
    assert status_payload['status'] == 'finished'
    assert status_payload['progress'] == 100

    # 获取任务结果
    result_resp = client.get(f'/get_result?task_id={task_id}')
    assert result_resp.status_code == 200
    result_payload = result_resp.get_json()
    assert result_payload['status'] == 'ok'
    assert result_payload['result'] == stub_training
    assert 'dashboard_url' in result_payload


if __name__ == "__main__":
    import pytest as _pytest

    raise SystemExit(_pytest.main([__file__]))

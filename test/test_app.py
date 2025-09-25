import os
import pytest
import json
from app import app

# 测试上传功能
@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = './uploads'
    with app.test_client() as client:
        yield client

# 测试上传文件
def test_upload(client):
    # 测试文件上传
    with open('test_files/test_forecast.xlsx', 'rb') as f1, \
         open('test_files/test_health.xlsx', 'rb') as f2, \
         open('test_files/test_monthly.xlsx', 'rb') as f3:
        data = {
            'fileForecast': (f1, 'test_forecast.xlsx'),
            'fileHealth': (f2, 'test_health.xlsx'),
            'fileMonthly': (f3, 'test_monthly.xlsx')
        }
        response = client.post('/upload_preview', data=data, follow_redirects=True)
        assert response.status_code == 200
        json_data = json.loads(response.data)
        assert json_data['status'] == 'ok'

# 测试预测任务启动
def test_start_task(client):
    with open('test_files/test_forecast.xlsx', 'rb') as f1, \
         open('test_files/test_health.xlsx', 'rb') as f2, \
         open('test_files/test_monthly.xlsx', 'rb') as f3:
        data = {
            'fileForecast': (f1, 'test_forecast.xlsx'),
            'fileHealth': (f2, 'test_health.xlsx'),
            'fileMonthly': (f3, 'test_monthly.xlsx')
        }
        response = client.post('/start_task', data=data, follow_redirects=True)
        assert response.status_code == 200
        json_data = json.loads(response.data)
        assert json_data['status'] == 'ok'
        task_id = json_data['task_id']
        assert task_id is not None

# 测试任务状态查询
def test_task_status(client):
    # 假设任务 ID 为 'dummy_task_id'
    task_id = 'dummy_task_id'
    response = client.get(f'/task_status?task_id={task_id}')
    assert response.status_code == 200
    json_data = json.loads(response.data)
    assert json_data['status'] in ['running', 'finished', 'failed']

# 测试获取预测结果
def test_get_result(client):
    task_id = 'dummy_task_id'
    response = client.get(f'/get_result?task_id={task_id}')
    assert response.status_code == 200
    json_data = json.loads(response.data)
    assert 'result' in json_data
    assert json_data['status'] == 'ok'

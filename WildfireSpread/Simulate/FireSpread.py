import os
import sys
import copy
import os.path
import logging
import threading
from dotenv import load_dotenv
from FireCore import fire_simulate
from flask import Flask, request, jsonify


logging.basicConfig(
    level=logging.WARNING,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logging.getLogger('FireCore').setLevel(logging.INFO)
logging.getLogger('FireUtils').setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = Flask(__name__)

simulation_tasks = {}
tasks_lock = threading.RLock()


def _get_json_body():
    """ 安全获取 JSON 请求体 """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({'Error': '请求体必须是合法JSON对象'}), 400)
    return data, None


def _get_task_id(data):
    """ 安全解析 fire_area_id """
    if 'fire_area_id' not in data:
        return None, (jsonify({'Error': '缺少fire_area_id'}), 400)

    try:
        task_id = int(data['fire_area_id'])
    except (ValueError, TypeError):
        return None, (jsonify({'Error': 'fire_area_id必须为整数.'}), 400)

    return task_id, None


def _build_thread(params, stop_event):
    """ 创建模拟线程，线程内使用参数副本，避免共享对象并发修改 """
    thread_params = copy.deepcopy(params)
    return threading.Thread(
        target=fire_simulate,
        args=(thread_params, stop_event),
        daemon=True,
        name=f'fire-task-{thread_params.get("fire_area_id", "unknown")}'
    )


@app.route('/fire_spread_simulation', methods=['POST'])
def simulate_api():
    try:
        global simulation_tasks

        request_initial, err = _get_json_body()
        if err:
            return err

        task_id, err = _get_task_id(request_initial)
        if err:
            return err

        logger.info('received simulation request.')
        logger.info('task id: %s, parameters: %s.', task_id, request_initial)

        with tasks_lock:
            # 如果已有任务，先中断
            if task_id in simulation_tasks:
                old_stop = simulation_tasks[task_id]['stop_event']
                old_stop.set()

                old_thread = simulation_tasks[task_id]['thread']
                if old_thread and old_thread.is_alive():
                    old_thread.join(timeout=3.0)
                    if old_thread.is_alive():
                        return jsonify({'Error': f'任务{task_id}旧线程尚未退出，请稍后重试.'}), 409

            # 创建新的 stop_event 和线程
            stop_event = threading.Event()
            thread = _build_thread(request_initial, stop_event)

            simulation_tasks[task_id] = {
                'params': copy.deepcopy(request_initial),
                'thread': thread,
                'stop_event': stop_event
            }
            thread.start()

        return jsonify({'Message': f'任务{task_id}开始模拟.'})

    except Exception as e:
        logger.exception('simulate_api error')
        return jsonify({'Error': str(e)}), 500


@app.route('/fire_spread_update', methods=['POST'])
def update_api():
    try:
        global simulation_tasks

        request_update, err = _get_json_body()
        if err:
            return err

        task_id, err = _get_task_id(request_update)
        if err:
            return err

        logger.info('received update request.')
        logger.info('task id: %s, parameters: %s.', task_id, request_update)

        with tasks_lock:
            if task_id not in simulation_tasks:
                return jsonify({'Error': f'任务{task_id}不存在，请调用/fire_spread_simulation初始化.'}), 400

            new_params = copy.deepcopy(simulation_tasks[task_id]['params'])
            updates = {
                k: v for k, v in request_update.items()
                if k != 'fire_area_id' and v is not None
            }
            new_params.update(updates)

            # 中断旧线程
            old_stop = simulation_tasks[task_id]['stop_event']
            old_stop.set()

            old_thread = simulation_tasks[task_id]['thread']
            if old_thread and old_thread.is_alive():
                old_thread.join(timeout=3.0)
                if old_thread.is_alive():
                    return jsonify({'Error': f'任务{task_id}旧线程尚未退出，请稍后重试.'}), 409

            # 启动新的模拟线程
            stop_event = threading.Event()
            thread = _build_thread(new_params, stop_event)

            simulation_tasks[task_id] = {
                'params': copy.deepcopy(new_params),
                'thread': thread,
                'stop_event': stop_event
            }
            thread.start()

        return jsonify({'Message': f'任务{task_id}参数已更新并重新模拟.'})

    except Exception as e:
        logger.exception('update_api error')
        return jsonify({'Error': str(e)}), 500


def get_app_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    try:
        app_dir = get_app_dir()
        config_path = os.path.join(app_dir, 'config.env')
        load_dotenv(config_path)

        ip_address = os.getenv('IP_ADDRESS', '127.0.0.1')
        port = int(os.getenv('PORT', 5000))
        debug = os.getenv('DEBUG', 'False').lower() == 'true'

        app.run(host=ip_address, port=port, debug=False)

    except Exception as e:
        print(f"模拟程序无法执行: {e}")

"""pandas 并行 apply 的真实 Jupyter kernel 中断测试。"""

import os
import time

import pytest


jupyter_client = pytest.importorskip("jupyter_client")
pytest.importorskip("ipykernel")


def _wait_for_iopub(client, message_id, *, timeout):
    """收集指定执行请求的 IOPub 消息，直到其回到 idle。"""
    deadline = time.monotonic() + timeout
    messages = []
    while time.monotonic() < deadline:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            message = client.get_iopub_msg(timeout=min(1.0, remaining))
        except Exception:
            continue
        if message.get("parent_header", {}).get("msg_id") != message_id:
            continue
        messages.append(message)
        if message.get("msg_type") == "status" and message.get("content", {}).get("execution_state") == "idle":
            return messages
    raise AssertionError(f"kernel 在 {timeout} 秒内未回到 idle")


def _execute(client, code, *, timeout=20):
    message_id = client.execute(code)
    messages = _wait_for_iopub(client, message_id, timeout=timeout)
    _wait_for_shell_reply(client, message_id, timeout=timeout)
    return message_id, messages


def _wait_for_shell_reply(client, message_id, *, timeout):
    """读取指定请求的 execute_reply，避免把旧请求回复误判为当前状态。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            message = client.get_shell_msg(timeout=1.0)
        except Exception:
            continue
        if message.get("parent_header", {}).get("msg_id") == message_id:
            return message
    raise AssertionError(f"kernel 在 {timeout} 秒内未返回 execute_reply")


@pytest.mark.integration
def test_interrupt_aborts_apply_without_restarting_kernel_or_losing_variables():
    """中断并行 apply 不能结束 kernel，也不能清空中断前的用户变量。"""
    manager = jupyter_client.KernelManager(kernel_name="python3")
    # Windows 使用 Jupyter 专用 interrupt event；独立进程组避免 kernel
    # 或其 loky 子进程的控制台事件影响 pytest 宿主。
    manager.start_kernel(cwd=os.getcwd(), independent=os.name == "nt")
    client = manager.blocking_client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=30)
        _, setup_messages = _execute(
            client,
            "import os, pandas as pd, hscredit\n"
            "中断前变量 = '仍然存在'\n"
            "中断前PID = os.getpid()\n"
            "print(f'{中断前PID}|{中断前变量}')",
        )
        setup_stream = "".join(
            message["content"].get("text", "") for message in setup_messages if message.get("msg_type") == "stream"
        )
        assert "仍然存在" in setup_stream

        long_message_id = client.execute(
            "from time import sleep\n"
            "def _触发中断(value):\n"
            "    if value == 0:\n"
            "        sleep(2)\n"
            "        raise KeyboardInterrupt('用户中断')\n"
            "    sleep(30)\n"
            "    return value\n"
            "pd.Series(range(8)).hscredit(n_jobs=2, bar=False, parallel_backend='loky').apply(_触发中断)"
        )
        interrupted_messages = _wait_for_iopub(client, long_message_id, timeout=20)
        interrupt_reply = _wait_for_shell_reply(client, long_message_id, timeout=10)
        interrupt_errors = [
            message.get("content", {}) for message in interrupted_messages if message.get("msg_type") == "error"
        ]
        assert interrupt_reply.get("content", {}).get("status") == "error", interrupt_reply.get("content")
        assert interrupt_reply.get("content", {}).get("ename") == "KeyboardInterrupt", interrupt_reply.get("content")
        assert not interrupt_errors or any(
            message.get("msg_type") == "error" and message.get("content", {}).get("ename") == "KeyboardInterrupt"
            for message in interrupted_messages
        ), interrupt_errors

        _, recovery_messages = _execute(
            client,
            "print(f'{os.getpid()}|{中断前PID}|{中断前变量}')",
            timeout=10,
        )
        recovery_stream = "".join(
            message["content"].get("text", "") for message in recovery_messages if message.get("msg_type") == "stream"
        )
        current_pid, original_pid, sentinel = recovery_stream.strip().split("|", 2)
        assert current_pid == original_pid
        assert sentinel == "仍然存在"
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)

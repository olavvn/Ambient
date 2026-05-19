"""스트리밍 엔진 테스트"""
import pytest
import asyncio
import torch
import threading
import time
import numpy as np
from src.streaming.buffer import ThemeBuffer, TokenQueue
from src.streaming.crossfade import crossfade_audio


def test_theme_buffer_update_get():
    buf = ThemeBuffer()
    tokens = torch.tensor([1, 2, 3, 4, 5])
    buf.update(tokens)
    result = buf.get()
    assert result is not None
    assert torch.equal(result, tokens)


def test_theme_buffer_new_flag():
    buf = ThemeBuffer()
    assert not buf.consume_new_flag()
    buf.update(torch.tensor([1]))
    assert buf.consume_new_flag()
    assert not buf.consume_new_flag()


def test_theme_buffer_thread_safety():
    """여러 스레드에서 동시 업데이트해도 안전"""
    buf = ThemeBuffer()
    errors = []

    def updater():
        try:
            for _ in range(100):
                buf.update(torch.randint(0, 100, (10,)))
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=updater) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_crossfade_shape():
    audio_out = np.random.randn(44100).astype(np.float32)
    audio_in = np.random.randn(44100).astype(np.float32)
    result = crossfade_audio(audio_out, audio_in, fade_samples=4410)
    assert result.shape == (4410,)
    assert result.dtype == np.float32


def test_crossfade_short_audio():
    """입력이 fade_samples보다 짧아도 크래시 없어야 함"""
    audio_out = np.random.randn(100).astype(np.float32)
    audio_in = np.random.randn(200).astype(np.float32)
    result = crossfade_audio(audio_out, audio_in, fade_samples=4410)
    assert result.dtype == np.float32
    assert len(result) > 0


@pytest.mark.asyncio
async def test_token_queue_put_get():
    loop = asyncio.get_event_loop()
    queue = TokenQueue(maxsize=4)
    queue.set_loop(loop)

    def put_in_thread():
        time.sleep(0.05)
        queue.put_nowait_threadsafe([1, 2, 3])

    threading.Thread(target=put_in_thread, daemon=True).start()
    chunk = await asyncio.wait_for(queue.get(), timeout=2.0)
    assert chunk == [1, 2, 3]

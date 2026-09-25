"""Railway entrypoint for Lara - Zen Etre Harmonie.

Starts one LiveAvatar LITE session, connects the LiveKit voice agent
directly to the LiveAvatar-hosted room, and shuts everything down cleanly.

This file intentionally avoids AgentServer/simulate_job because the
LiveKit room is provisioned by LiveAvatar.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import utils
from livekit.plugins import silero

from agent import LiveAvatarAgent
from avatar_ws import AvatarWebSocket
from liveavatar_client import LiveAvatarClient
from pipeline import (
    build_room_options,
    build_session,
    mute_agent_audio_on_publish,
    wire_room_observability,
    wire_session_observability,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lara-railway")

load_dotenv(".env.local")


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


async def main() -> None:
    api_key = os.environ["LIVEAVATAR_API_KEY"]
    avatar_id = os.environ["AVATAR_ID"]
    is_sandbox = env_bool("IS_SANDBOX", True)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    room = rtc.Room()
    avatar_ws: AvatarWebSocket | None = None
    started = None

    async with utils.http_context.open():
        async with LiveAvatarClient(api_key=api_key) as liveavatar:
            try:
                logger.info("Creating Lara LiveAvatar session...")

                token = await liveavatar.create_session_token(
                    avatar_id=avatar_id,
                    is_sandbox=is_sandbox,
                )

                started = await liveavatar.start_session(
                    session_token=token.session_token
                )

                logger.info(
                    "LiveAvatar session started session_id=%s",
                    started.session_id,
                )

                wire_room_observability(room)
                mute_agent_audio_on_publish(room)

                logger.info("Connecting Lara agent to LiveKit room...")
                await room.connect(
                    started.livekit_url,
                    started.livekit_agent_token,
                )

                avatar_ws = AvatarWebSocket(ws_url=started.ws_url)
                await avatar_ws.connect()

                vad = silero.VAD.load()
                session = build_session(vad)
                wire_session_observability(session)

                await session.start(
                    agent=LiveAvatarAgent(avatar_ws=avatar_ws),
                    room=room,
                    room_options=build_room_options(),
                )

                logger.info("Lara is connected and ready.")

                disconnected = asyncio.Event()

                @room.on("disconnected")
                def _on_disconnected(*_args) -> None:
                    disconnected.set()

                stop_task = asyncio.create_task(stop.wait())
                disconnect_task = asyncio.create_task(disconnected.wait())

                done, pending = await asyncio.wait(
                    {stop_task, disconnect_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()

                for task in done:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

            finally:
                logger.info("Stopping Lara...")

                if avatar_ws is not None:
                    with contextlib.suppress(Exception):
                        await avatar_ws.close()

                if room.isconnected():
                    with contextlib.suppress(Exception):
                        await room.disconnect()

                if started is not None:
                    with contextlib.suppress(Exception):
                        await liveavatar.stop_session(
                            session_id=started.session_id
                        )

                logger.info("Lara stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())

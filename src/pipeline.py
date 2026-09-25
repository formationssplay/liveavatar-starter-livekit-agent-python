"""Shared voice pipeline + room helpers.

Used by both demos and by the Railway direct-mode entrypoint.

Railway direct mode uses Deepgram directly for STT and TTS so it does not
depend on LiveKit Inference authentication for the LiveAvatar-hosted room.
"""

from __future__ import annotations

import logging

from livekit import rtc
from livekit.agents import AgentSession, inference, room_io
from livekit.plugins import ai_coustics, deepgram


logger = logging.getLogger("pipeline")

AGENT_MODEL = "openai/gpt-5.3-chat-latest"


def build_session(vad) -> AgentSession:
    """Direct Deepgram STT + TTS with VAD.

    The LLM generation path is overridden by LiveAvatarAgent.llm_node,
    which sends the user's request to the n8n Agent Directeur.
    """
    return AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="fr",
        ),
        llm=inference.LLM(model=AGENT_MODEL),
        tts=deepgram.TTS(
            model="aura-2-agathe-fr",
        ),
        turn_detection=None,
        vad=vad,
        preemptive_generation=True,
    )


def build_room_options() -> room_io.RoomOptions:
    """Audio input options.

    Output is left enabled so AgentSession runs the TTS pipeline and the
    agent's tts_node override fires. The raw published track is muted so
    the room does not carry double audio because the avatar publishes the
    lip-synced voice.
    """
    return room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=ai_coustics.BVC(),
        ),
    )


def mute_agent_audio_on_publish(room: rtc.Room) -> None:
    @room.on("local_track_published")
    def _on_local_track_published(pub, track):
        if pub.kind == rtc.TrackKind.KIND_AUDIO and isinstance(
            track, rtc.LocalAudioTrack
        ):
            track.mute()
            logger.info(
                "muted local agent audio track sid=%s",
                pub.sid,
            )


def wire_room_observability(room: rtc.Room) -> None:
    @room.on("participant_connected")
    def _on_participant_connected(participant):
        logger.info(
            "participant_connected identity=%s kind=%s",
            participant.identity,
            participant.kind,
        )

    @room.on("participant_disconnected")
    def _on_participant_disconnected(participant):
        logger.info(
            "participant_disconnected identity=%s",
            participant.identity,
        )

    @room.on("track_subscribed")
    def _on_track_subscribed(track, publication, participant):
        logger.info(
            "track_subscribed kind=%s sid=%s from=%s",
            track.kind,
            publication.sid,
            participant.identity,
        )

    @room.on("disconnected")
    def _on_disconnected(*_args):
        logger.info("room disconnected")


def wire_session_observability(session: AgentSession) -> None:
    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(event):
        logger.info(
            "user_input_transcribed final=%s transcript=%s",
            event.is_final,
            event.transcript,
        )

    @session.on("agent_state_changed")
    def _on_agent_state_changed(event):
        logger.info(
            "agent_state %s -> %s",
            event.old_state,
            event.new_state,
        )

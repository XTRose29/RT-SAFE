#!/usr/bin/env python3
"""Optional owner-controlled /rtsafe bot for the configured Slack channel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading

import model_watch as watch


def authorized(body, settings):
    user = body.get("user_id") or (body.get("user") or {}).get("id")
    channel = body.get("channel_id") or (body.get("channel") or {}).get("id")
    return bool(settings.get("owner_user_id") and settings.get("channel_id")
                and user == settings["owner_user_id"] and channel == settings["channel_id"])


def blocks(row):
    billing = ("Uses pooled subscription quota; paid API fallback is disabled."
               if row.get("provider") == "account-pool" else
               f"Planning estimate: ${row['planning_estimate_usd']:.2f}. "
               "This is a token-use estimate, not a spending cap; retries can add cost.")
    text = (f"RT-SAFE: {row['model']} — {row['state']}\n"
            f"{row['episodes']} episodes. {billing}")
    result = [{"type": "section", "text": {"type": "plain_text", "text": text}}]
    if row["state"] in {"queued", "needs_attention"}:
        result.append({"type": "actions", "elements": [{"type": "button", "action_id": "rtsafe_start",
                       "text": {"type": "plain_text", "text": "Start / resume"}, "value": row["id"]}]})
    elif row["state"] == "running":
        result.append({"type": "actions", "elements": [{"type": "button", "action_id": "rtsafe_stop",
                       "text": {"type": "plain_text", "text": "Stop"}, "value": row["id"]}]})
    return result


def load_tokens(path):
    path = Path(path)
    if path.stat().st_mode & 0o077:
        raise ValueError("Slack token file must have mode 600")
    tokens = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        tokens[key.strip()] = value.strip().strip("\"'")
    for key, prefix in (("SLACK_BOT_TOKEN", "xoxb-"), ("SLACK_APP_TOKEN", "xapp-")):
        if not tokens.get(key, "").startswith(prefix):
            raise ValueError(f"Missing or invalid {key}")
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = watch.read(args.config)
    settings = config["slack"]
    if not settings.get("owner_user_id") or not settings.get("channel_id"):
        parser.error("Set slack.owner_user_id and slack.channel_id before connecting")
    tokens = load_tokens(settings["token_file"])
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler
    app = App(token=tokens["SLACK_BOT_TOKEN"])
    state = Path(config["state_dir"])

    def command(*parts):
        return [config["python"], str(watch.ROOT / "tools/model_watch.py"),
                "--config", str(args.config.resolve()), *parts]

    @app.command("/rtsafe")
    def menu(ack, body, respond):
        ack()
        if body.get("channel_id") != settings["channel_id"]:
            respond(text="Use /rtsafe in the configured benchmark channel.", response_type="ephemeral")
            return
        rows = watch.status(config)
        lines = [f"{r['model']}: {r['state']} ({r['episodes']} episodes)" for r in rows]
        respond(text="\n".join(lines) or "No queued models yet.", response_type="ephemeral")

    @app.action("rtsafe_start")
    def start(ack, body, respond):
        ack()
        if not authorized(body, settings):
            respond(text="Only the configured RT-SAFE owner can start runs in this channel.", response_type="ephemeral", replace_original=False)
            return
        job_id = body["actions"][0]["value"]
        path = watch.job_path(config, job_id)
        # The detached CLI owns the campaign lock and remains alive if Slack disconnects.
        with (path.parent / "launcher.log").open("a") as stream:
            subprocess.Popen(command("start", job_id), stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        respond(text="Start requested. The next status update reports whether it started; an existing campaign keeps its lock.",
                response_type="ephemeral", replace_original=False)

    @app.action("rtsafe_stop")
    def stop(ack, body, respond):
        ack()
        if not authorized(body, settings):
            respond(text="Only the configured owner can stop runs.", response_type="ephemeral", replace_original=False)
            return
        try:
            watch.stop_job(config, body["actions"][0]["value"])
            text = "Stop requested. Completed episodes are preserved for resume."
        except Exception as exc:
            text = str(exc)
        respond(text=text, response_type="ephemeral", replace_original=False)

    def notify():
        checkpoint = state / "slack_notifications.json"
        sent = watch.read(checkpoint) if checkpoint.exists() else {}
        release_checkpoint = state / "slack_release_notifications.json"
        release_file = state / "releases.json"
        # Do not flood a newly connected channel with the entire historical feed.
        releases_sent = (watch.read(release_checkpoint) if release_checkpoint.exists()
                         else {name: True for name in (watch.read(release_file) if release_file.exists() else {})})
        watch.write(release_checkpoint, releases_sent)
        while True:
            try:
                releases = watch.read(release_file) if release_file.exists() else {}
                for name in releases:
                    if name not in releases_sent:
                        app.client.chat_postMessage(channel=settings["channel_id"],
                            text=f"New model detected: {name}. RT-SAFE requires verified vision input, an authorized account route and reasoning settings before Start is available.")
                        releases_sent[name] = True
                        watch.write(release_checkpoint, releases_sent)
                for row in watch.status(config):
                    version = row["state"] + ":" + row.get("updated", "")
                    if sent.get(row["id"]) == version:
                        continue
                    message = blocks(row)
                    if row["state"] == "completed":
                        result = watch.read(watch.job_path(config, row["id"]).parent / "result.json")
                        details = "\n".join(f"{r['label']}: success {r['success']:.1f}%, safe success {r['safeSuccess']:.1f}%, collisions {r['collisions']:.2f}" for r in result["effort"])
                        message.append({"type": "section", "text": {"type": "plain_text", "text": details}})
                    app.client.chat_postMessage(channel=settings["channel_id"], text=f"RT-SAFE {row['model']}: {row['state']}", blocks=message)
                    sent[row["id"]] = version
                    watch.write(checkpoint, sent)
            except Exception as exc:
                # Avoid logging provider responses or credentials.
                print(f"Notification deferred: {type(exc).__name__}", file=sys.stderr, flush=True)
            threading.Event().wait(15)

    with watch.lock(state / "slack.lock", blocking=False):
        threading.Thread(target=notify, daemon=True).start()
        SocketModeHandler(app, tokens["SLACK_APP_TOKEN"]).start()


if __name__ == "__main__":
    main()

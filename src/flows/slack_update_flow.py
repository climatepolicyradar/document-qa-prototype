#  type: ignore
"""
Flows to update the slack channel with the latest status of running experiments.

We pyright ignore the whole file because the prefect client isn't typed properly for it
"""
from prefect import flow
from prefect import get_run_logger
from prefect.server.schemas.filters import FlowFilter, FlowRunFilter
from src.flows.utils import get_db, send_slack_message
from prefect import get_client
from prefect.client.schemas.objects import StateType
from datetime import datetime, timedelta
import asyncio


def get_experiment_queue_status(tag: str):
    db = get_db()
    null_cursor = db.execute_sql(
        f"SELECT count(*) FROM public.queue WHERE dequeued_at is null AND q_name = '{tag}'"
    )
    not_null_cursor = db.execute_sql(
        f"SELECT count(*) FROM public.queue WHERE dequeued_at is not null AND q_name = '{tag}'"
    )

    return {
        "to_process": null_cursor.fetchone()[0],
        "processed": not_null_cursor.fetchone()[0],
    }


async def get_flow_stats(flow_names: list[str]):
    async with get_client() as client:
        for flow_name in flow_names:
            flows = await client.read_flows(
                flow_filter=FlowFilter(name={"any_": [flow_name]})
            )
            if not flows:
                print(f"No flow found with name: {flow_name}")
                continue

            # Get current time and time 1 hour ago
            now = datetime.utcnow()
            one_hour_ago = now - timedelta(hours=1)

            # Get running flow runs
            running_runs = await client.read_flow_runs(
                flow_filter=FlowFilter(name={"any_": [flow_name]}),
                flow_run_filter=FlowRunFilter(
                    state={"type": {"any_": [StateType.RUNNING]}}
                ),
            )
            running_count = len(running_runs)

            # Get scheduled flow runs
            scheduled_runs = await client.read_flow_runs(
                flow_filter=FlowFilter(name={"any_": [flow_name]}),
                flow_run_filter=FlowRunFilter(
                    state={"type": {"any_": [StateType.SCHEDULED]}}
                ),
            )
            scheduled_count = len(scheduled_runs)

            # Get failed flow runs in the last hour
            failed_runs = await client.read_flow_runs(
                flow_filter=FlowFilter(name={"any_": [flow_name]}),
                flow_run_filter=FlowRunFilter(
                    state={"type": {"any_": [StateType.FAILED]}},
                    start_time={"after_": one_hour_ago.isoformat()},
                ),
            )
            failed_count = len(failed_runs)

            send_slack_message(
                f"Flow: {flow_name}: 💚 Currently running: {running_count}, 📅 Scheduled: {scheduled_count}, 🔴😭 Failed in the last hour: {failed_count}",
                include_flow_url=False,
            )


@flow
async def generate_update(
    tags: list[str] = [
        "g_eval_comparison_experiment_2",
        "main_experiment_run_2024_08_24",
    ],
):
    logger = get_run_logger()
    logger.info(f"Generating update for tags {tags}")
    send_slack_message(
        f"🧮 Here is your experiment update as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        include_flow_url=False,
    )

    for tag in tags:
        status = get_experiment_queue_status(tag)
        send_slack_message(
            f"🤔 Queue for tag {tag} has {status['to_process']} to process and {status['processed']} processed",
            include_flow_url=False,
        )

    await get_flow_stats(
        [
            "process-answer-job-from-queue",
            "queue-answer-flow",
            "process-eval-experiment-from-queue",
        ]
    )


if __name__ == "__main__":
    asyncio.run(generate_update())

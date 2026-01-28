import redis
import wandb
import json
import time
import os

def recover_logs(redis_url="redis://localhost:6379", project="pilot_flwr"):
    print(f"Connecting to Redis at {redis_url}")
    r = redis.from_url(redis_url, decode_responses=True)
    
    keys = r.keys("logs:*")
    if not keys:
        print("No log keys found in Redis.")
        return

    print(f"Found log keys: {keys}")

    # Initialize wandb
    run_name = f"recovered-logs-{int(time.time())}"
    print(f"Initializing WandB run: {run_name} in project {project}")
    wandb.init(project=project, name=run_name)

    total_logs = 0
    
    for key in keys:
        client_name = key.split(":")[-1]
        print(f"Processing logs for {client_name}...")
        
        # Read all logs without deleting
        log_entries_str = r.lrange(key, 0, -1)
        print(f"Found {len(log_entries_str)} entries.")
        
        for log_entry_str in log_entries_str:
            try:
                log_entry = json.loads(log_entry_str)
                step = log_entry.pop("step", None)
                if step is not None:
                    log_entry["client_step"] = step
                    # Add client name to differentiate if multiple clients
                    log_entry["client_name"] = client_name
                    wandb.log(log_entry)
                    total_logs += 1
            except json.JSONDecodeError:
                print(f"Failed to decode log entry: {log_entry_str}")

    print(f"Finished. Logged {total_logs} entries to WandB.")
    wandb.finish()

if __name__ == "__main__":
    recover_logs()

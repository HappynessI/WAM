# WAM Agent Notes

## Repository

- GitHub: `git@github.com:HappynessI/WAM.git`
- Main local workspace: `/data0/code/WAM`
- Active policy repository: `/data0/code/WAM/frappe-main`

## Development And Evaluation

Current development and local RoboTwin evaluation are performed on this machine. Use this machine to edit code, run smoke tests, and validate single-task evaluation chains before packaging.

## Training

Formal training is expected to run on a separate offline H200 machine. Prepare self-contained sync packages from this workspace before transferring to the H200 environment.

## Current Evaluation Context

- Local RoboTwin root: `/data0/code/RoboTwin`
- FRAPPE eval entry: `/data0/code/WAM/frappe-main/eval.sh`
- Local encoder weights: `/data0/code/WAM/weights/RDT`
- Current packaged bundle: `/data0/code/WAM/wam_heatmap_adjust_bottle_sync_20260524`

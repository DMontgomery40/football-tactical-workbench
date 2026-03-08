# Football Tactical Workbench

A local Mac-friendly workbench for wide-angle football analysis with a real UI, live model preview, and automatic pitch calibration.

## What it does

- React UI on **http://127.0.0.1:4317**
- FastAPI backend on **http://127.0.0.1:8431**
- Football-specific detection with **Soccana** weights
- Player, ball, and referee detection
- ByteTrack-based multi-object tracking
- Unsupervised home/away separation from jersey colors
- Automatic field-keypoint calibration with **Soccana_Keypoint**
- Pitch projection refreshed every **10 frames**
- Live model preview in the browser while the clip plays
- Saved overlay video plus CSV diagnostics and summaries

## Current model stack

- Detector: `backend/models/soccana/Model/weights/best.pt`
- Field calibration: `backend/models/soccana_keypoint/Model/weights/best.pt`

These are local `.pt` files. The backend can also resolve them from Hugging Face if missing. Local model weights and cached downloads are not meant to be committed to git.

## What the UI is for

The UI is the primary debugging surface.

You should be able to tell from the browser:

- whether player tracking is stable
- whether the ball detector is firing or hallucinating
- whether team clustering looks plausible
- whether field calibration is locking and staying locked
- whether the projected minimap matches the play

This repo is not meant to be “terminal only”.

## Screenshots

Overview:

![Workbench dashboard](docs/screenshots/workbench-dashboard.png)

Run review:

![Run review with experimental signal, overlay, and diagnostics](docs/screenshots/workbench-run-review.png)

## Fastest way to run

### One command

```bash
echo "starting workbench" && cd . && bash run_all.sh
```

### Two terminal method

Terminal 1:

```bash
echo "starting backend" && cd backend && bash run_backend.sh
```

Terminal 2:

```bash
echo "starting frontend" && cd frontend && bash run_frontend.sh
```

## Typical workflow

1. Load a clip from disk or upload one in the UI.
2. Click `Load clip`.
3. Start `Live model preview`.
4. Watch the browser overlay for:
   - player and ball boxes
   - team labels
   - calibration status
   - minimap behavior
5. If the preview looks sane, click `Run tactical demo`.
6. Review the saved overlay and diagnostics after completion.

## Good first use

Start with a short clip first.

- 10 to 30 seconds
- one broadcast camera phase
- wide-angle view with visible field structure
- minimal replay cuts

Full matches are supported, but they are long-running jobs and should be treated like batch work.

## Where to point it

You have two choices in the UI:

1. upload a video file
2. paste a local path to a video on your Mac

You can also scan a dataset folder and click one of the discovered videos to auto-fill the path field.

## Common local data locations

- Bundesliga sample clips: `backend/datasets/bundesliga_sample`
- YouTube match downloads: `backend/datasets/youtube_clips`
- SoccerNet workspace root: `backend/datasets/soccernet`

These are local working directories and are not meant to be committed to git.
The repository does **not** ship SoccerNet footage.

## SoccerNet access

SoccerNet video access is not public in the normal sense.

- This repository provides the platform and tooling to work with SoccerNet data.
- This repository does **not** provide, bundle, or redistribute SoccerNet videos or labels.
- It is password protected.
- You must have your **own** SoccerNet NDA-approved access.
- You must use your **own** SoccerNet password/credentials.
- Access is tied to the SoccerNet terms you personally agreed to, including non-commercial usage restrictions.

Do not share downloaded SoccerNet videos or credentials through this repo.

## Acknowledgements

- Bundesliga clip samples used during UI and pipeline iteration came from `dbal0503/BundesLiga` on Hugging Face.
- Detection and pitch-keypoint workbench defaults are built around Adit Jain's `soccana` and `Soccana_Keypoint` weights on Hugging Face.
- SoccerNet provides the password-protected match video and event-label ecosystem that makes the goal-aligned experiment possible for approved users.

## Experiments

This workbench is meant to help people run their own football experiments on top of a stable detection, tracking, and pitch-calibration pipeline.

The current repository includes one example experiment: a goal-aligned **geometric volatility index** built from projected player positions on the pitch. It is not the product definition and it is not the only intended use of the repo. It is just one concrete example of the kind of analysis you can layer on top once the wide-angle pipeline is working.

In plain language, the current experiment asks a simple question:

- how much are team shape and pitch occupation changing from second to second
- does that volatility rise in the build-up to goals

If you want to use the workbench for different ideas, that is the point. You can swap in different labels, different match-state features, different summaries, or a completely different downstream task.

## Example Experiment Math

The current example experiment samples projected player positions at **1 Hz**, summarizes team shape and whole-pitch occupation, measures how quickly those values change, and then aligns the resulting signal against goal timestamps.

### Per-team, per-second features

For each team with projected player positions $p_i = (x_i, y_i)$:

- Centroid:

$$
\bar{p}_t = \frac{1}{n}\sum_{i=1}^{n} p_i
$$

- Spread RMS:

$$
S_t = \sqrt{\frac{1}{n}\sum_{i=1}^{n} \|p_i - \bar{p}_t\|^2}
$$

- Team length along the pitch axis:

$$
L_t = Q_{0.90}(x) - Q_{0.10}(x)
$$

- Team width across the pitch:

$$
W_t = Q_{0.90}(y) - Q_{0.10}(y)
$$

where $Q_q$ is the empirical quantile, which is more robust than a raw max-minus-min range.

### Match-state features

- Inter-team centroid distance:

$$
D_t = \|\bar{p}^{(home)}_t - \bar{p}^{(away)}_t\|
$$

- Spatial entropy over a fixed pitch grid:

$$
H_t = -\sum_{j=1}^{k} p_j \log(p_j)
$$

where $p_j$ is the fraction of all projected players falling in grid cell $j$.

### Volatility features

The experiment samples the match state at **1 Hz**.

For each scalar feature series $x_t$, compute the absolute first difference:

$$
\Delta x_t = |x_t - x_{t-1}|
$$

Then compute a rolling 10-second mean absolute delta:

$$
V_x(t) = \frac{1}{w}\sum_{\tau=t-w+1}^{t} \Delta x_{\tau}
$$

with $w = 10$ seconds.

The current feature set includes:

- home spread volatility
- away spread volatility
- home length volatility
- away length volatility
- inter-team centroid-distance volatility
- spatial-entropy volatility

### Combined volatility index

Each volatility feature is z-scored within the half:

$$
Z_x(t) = \frac{V_x(t) - \mu_x}{\sigma_x}
$$

Then the combined volatility index is:

$$
\text{VolIndex}_t = \frac{1}{m}\sum_{x \in \mathcal{F}} Z_x(t)
$$

where $\mathcal{F}$ is the active feature set and $m$ is the number of finite z-scores at time $t$.

### Goal alignment

From SoccerNet `Labels-v2.json`, each goal event is aligned to the current half using the provided `position` timestamp in milliseconds.

For each second $t$, the experiment derives:

- `seconds_to_next_goal`
- `goal_in_next_30s`
- `goal_in_next_60s`

### Example validation metric

One simple way to evaluate the signal is the 30-second uplift:

$$
\text{Uplift}_{30s} =
\frac{
\mathbb{E}[\text{VolIndex}_t \mid \text{goal in next 30s}] -
\mathbb{E}[\text{VolIndex}_t \mid \text{no goal in next 30s}]
}{
\mathbb{E}[\text{VolIndex}_t \mid \text{no goal in next 30s}]
}
$$

## Stop the background backend

```bash
echo "stopping backend" && cd . && bash stop_backend.sh
```

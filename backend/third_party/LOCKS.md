# Soccer Benchmark Evaluator Locks

These are the pinned upstream references the benchmark adapters expect when vendored under `backend/third_party/soccernet/`.

| Project | Upstream | Pinned HEAD on 2026-03-12 |
| --- | --- | --- |
| `sskit` | https://github.com/Spiideo/sskit | `5085b7d7d76eebaeb35d4659bbca972031167268` |
| `sn-calibration` | https://github.com/SoccerNet/sn-calibration | `ab38f461bec729fead86b6986839de1bb826f16d` |
| `sn-tracking` | https://github.com/SoccerNet/sn-tracking | `b0bbba35e07ff58010b6313ef8aa59ef663ad392` |
| `sn-spotting` | https://github.com/SoccerNet/sn-spotting | `9842826f94e1419580a9d17219c11aca7225f7ce` |
| `sn-teamspotting` | https://github.com/SoccerNet/sn-teamspotting | `091fed2fc35c33f7489f3596958a2fe385e37d65` |
| `sn-gamestate` | https://github.com/SoccerNet/sn-gamestate | `057dd144a982e00576f8ffb45bdd00c0f614c549` |
| `FOOTPASS` | https://github.com/JeremieOchin/FOOTPASS | `8fd37077b076879e809f4d48da4c57aaed8a73e5` |
| `tracklab` | https://github.com/TrackingLaboratory/tracklab | `095306aa4bd89c94ead5c579b15c41870d1356f2` |

The adapters fail visibly when a required checkout is missing. Clone the projects into `backend/third_party/soccernet/<repo-name>/` at the pinned commit before running those suites.

# Performance: Python vs C++ (measured with tools/measure_load.py + tools/measure_topics.py)

CPU % are of ONE core (100 = one full core); 32 logical CPUs. Same machine, same demo (Webots, stereo, Nav2), same barrel seed.


## Car stationary (40 s window)

| metric (mean) | Python | C++ | change |
|---|---:|---:|---:|
| perception_node | 75.2 | 36.7 | -51 % |
| stamp_relay | 66.1 | 12.8 | -81 % |
| disparity_to_depth | 9.3 | 2.3 | -75 % |
| stereo SGBM container | 83.8 | 54.8 | -35 % |
| other offroad_nav2 nodes | 33.2 | 4.5 | -87 % |
| Nav2 (all servers) | 38.0 | 20.1 | -47 % |
| localization (ekf/navsat) | 9.6 | 5.2 | -46 % |
| Webots + drivers | 84.3 | 56.4 | -33 % |
| TOTAL host CPU (% of all cores) | 20.1 | 10.7 | -47 % |
| GPU util % | 29.9 | 34.4 | +15 % |
| GPU mem MiB | 1774.0 | 2214.0 | +25 % |
| GPU power W | 62.4 | 61.8 | -1 % |
| perception_node GPU SM % | 13.2 | 10.1 | -23 % |
| Webots GPU SM % | 5.1 | 11.2 | +119 % |

| end-to-end | Python | C++ |
|---|---:|---:|
| image-to-cloud latency p50 (ms) | 144.0 | 72.0 |
| image-to-cloud latency p95 (ms) | 648.0 | 96.0 |
| image-to-cloud latency max (ms) | 720.0 | 96.0 |
| cloud rate (Hz) | 7.2 | 9.47 |
| depth rate (Hz) | 8.2 | 9.45 |

## Car driving a 95 m goal at 3 m/s (45 s window)

| metric (mean) | Python | C++ | change |
|---|---:|---:|---:|
| perception_node | 84.1 | 38.0 | -55 % |
| stamp_relay | 62.9 | 12.8 | -80 % |
| disparity_to_depth | 8.8 | 2.4 | -73 % |
| stereo SGBM container | 85.5 | 59.1 | -31 % |
| other offroad_nav2 nodes | 31.1 | 4.8 | -85 % |
| Nav2 (all servers) | 39.3 | 23.5 | -40 % |
| localization (ekf/navsat) | 8.9 | 5.4 | -39 % |
| Webots + drivers | 80.8 | 57.1 | -29 % |
| TOTAL host CPU (% of all cores) | 20.1 | 8.9 | -56 % |
| GPU util % | 29.6 | 35.3 | +19 % |
| GPU mem MiB | 1774.0 | 2214.0 | +25 % |
| GPU power W | 67.6 | 63.4 | -6 % |
| perception_node GPU SM % | 16.0 | 11.0 | -31 % |
| Webots GPU SM % | 4.1 | 9.0 | +119 % |

| end-to-end | Python | C++ |
|---|---:|---:|
| image-to-cloud latency p50 (ms) | 216.0 | 72.0 |
| image-to-cloud latency p95 (ms) | 624.0 | 96.0 |
| image-to-cloud latency max (ms) | 792.0 | 144.0 |
| cloud rate (Hz) | 8.42 | 10.02 |
| depth rate (Hz) | 8.71 | 10.02 |

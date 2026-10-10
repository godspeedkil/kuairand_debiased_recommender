# Short-video recommender with debiased ranking (KuaiRand)

Building a retrieve → rank → re-rank recommender on the [KuaiRand](https://kuairand.com/) dataset.

> KuaiRand is an unbiased sequential recommendation dataset collected from the recommendation logs of the video-sharing mobile app, Kuaishou (快手). It is the first recommendation dataset with millions of intervened interactions of randomly exposed items inserted in the standard recommendation feeds!

### Why KuaiRand?
1. The randomly intervened nature of the dataset is perfect for showcasing a series of debiasing techniques.
2. Sequential logs also provide an opportunity to deal with leakage and other time-based considerations.
3. Tiered dataset (Pure vs 1K vs 27K) provides an appropriate proof of concept → testing → implementation → scaling pathway.
4. Rich, encrypted features allow for feature analysis and selection to be showcased, as well.

## Data

| Release | Users | Videos | Rows | Project Role |
| :------ | ----: | -----: | ---: | :-------- |
| Pure | 27,285 | 7,583 | 2.6M | Ranking and debiasing development |
| 1K | 1,000 | 4.4M | 11.8M | Content-based model on full user histories |
| 27K | 27,285 | 32M | 322M | Full-scale model, final numbers |

`notebooks/02_versions.ipynb` checks that Pure and 1K are exact filters of 27K with
shared IDs. Pure keeps every user's logs on the 7,583 videos in the random-exposure
pool, 1K keeps 1,000 users' full logs.

- **Splits:** train 04-08..04-30, val 05-01..05-04, test 05-05..05-08, on the `date`
  column. Events are ordered by (`date`, `time_ms`): about 0.7% of rows carry the next
  day's `date` for a late-evening `time_ms`.
- **Standard vs random logs:** the standard log holds the policy's recommendations.
  The random log holds only the videos inserted at random into the same feeds.
- **The pool is a specific slice:** 7,583 videos uploaded 04-09..04-11. Results on the random log describe these videos, not the catalog.
- **Val is holiday traffic** (2022 Chinese Labor Day). Check that models rank the same way on val and test before using val to choose between them.

## Evaluation

Models are fit on standard logs before each eval window, then evaluated two ways:

- **Retrieval:** rank the full catalog per user and compare the top K with their eval
  positives. It runs on standard logs only (the random log has unbiased labels just
  for pool videos), so it is a biased measure of the task.
- **Ranking (main task):** order each user's impressions and compare with what they
  clicked or watched (AUC, plus GAUC and NDCG with 95% bootstrap intervals over
  users).
  On the random log this is unbiased, since the policy didn't choose those videos.

## Baselines

Ranking on the random log, `is_click`, GAUC with 95% interval:

| model | Pure val | Pure test | 1K val | 1K test |
| :---- | -------: | --------: | -----: | ------: |
| most popular | 0.584 (0.580–0.587) | 0.582 (0.580–0.585) | 0.567 (0.548–0.583) | 0.572 (0.556–0.586) |
| recent popular (3 days) | 0.563 (0.560–0.566) | 0.561 (0.558–0.564) | 0.516 (0.506–0.527) | 0.525 (0.516–0.535) |
| item co-occurrence | 0.595 (0.592–0.599) | 0.594 (0.591–0.597) | 0.510 (0.503–0.517) | 0.515 (0.508–0.521) |
| random | 0.502 (0.499–0.505) | 0.501 (0.498–0.504) | 0.492 (0.476–0.508) | 0.501 (0.485–0.515) |

Collaborative baselines barely beat random on 1K: its 1,000 users give about 28x less
evidence per pool video than Pure, and its intervals are about 5x wider.

## Plan

1. **Pure:** a learned ranker on point-in-time features, then debiasing methods that
   use the random log's train window.
2. **1K:** a content-based model built from full user histories (tags, author,
   duration, etc.).
3. **27K:** the same models at full scale. 1K's users are in all three releases, so all
   models are compared on 1K users' random-log rows.

## Quickstart

```bash
make install                              # package plus dev tools
bash scripts/download_kuairand.sh Pure    # or 1K, 27K
krec ingest --config configs/pure.yaml    # or 1k.yaml, 27k.yaml
krec baselines --config configs/pure.yaml # tables in reports/*/baselines/
```

`krec all` runs ingest, features, and baselines. `krec profile` summarizes a release
for notebook analysis. `krec synth --config configs/synthetic.yaml` makes
a small synthetic dataset with planted exposure bias, for trying things without a
download.

## Development

```bash
make check   # ruff, Black, and the tests, as CI runs them
```

The tests run the whole pipeline and both notebooks on synthetic data, including a
mutation test for the feature-leakage guard. CI runs `make check` on Python 3.10 and
3.13.

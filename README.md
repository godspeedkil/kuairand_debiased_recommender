# Short-video recommender with debiased ranking (KuaiRand)

Building a retrieve → rank → re-rank recommender on the [KuaiRand](https://kuairand.com/) dataset.

> KuaiRand is an unbiased sequential recommendation dataset collected from the recommendation logs of the video-sharing mobile app, Kuaishou (快手). It is the first recommendation dataset with millions of intervened interactions of randomly exposed items inserted in the standard recommendation feeds!

### Why KuaiRand?
1. The randomly intervened nature of the dataset is perfect for showcasing a series of debiasing techniques.
2. Sequential logs also provide an opportunity to deal with leakage and other time-based considerations.
3. Tiered dataset (Pure vs 1K vs 27K) provides an appropriate proof of concept → testing → implementation → scaling pathway.
4. Rich, encrypted features allow for feature analysis and selection to be showcased, as well.

## Smoke tests

```bash
pip install -e ".[dev]"
pytest
```
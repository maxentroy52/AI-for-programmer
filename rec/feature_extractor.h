#pragma once

#include <thread>
#include <vector>

#include "feature.h"

// For online-feature computation.
// In this scenario, it's for computing cross features real time.
// But in our rs, it's also for computing user and item features. So, we need another data structure for storing the real-time computing features.
// You may wonder, why user and item features still needs online computing?
// 1. The direct reason is that it is the input param for fe lib, if you are trying to use fe lib, you must use the raw sample which means additional computing.
// 2. From my perspective, the advantage of using raw sample is that it is a unified data structure and used in both online and offline.
// In order to keep the consistency between online and offline, we should reduce the operation that will be performed in online and offline.
// If we use the RawSample, there will be no more concatenation operation in offline.
class FeatureExtractor {
 public:
  void FeatureExtract(const UserFeatures& user, const std::vector<ItemFeatures>& item_list, const ContextFeatures& ctx,
                      std::vector<CrossFeatures>& cross_list, std::vector<BehaviorFeatures> behavior_list) {
    int item_size = item_list.size();
    cross_list.resize(item_size);
    behavior_list.resize(item_size);

    for (int i = 0; i < item_size; ++i) {
      ComputeCrossFeatures(user, item_list[i], ctx, cross_list[i]);
      ComputeBehaviorFeatures(user, item_list[i], behavior_list[i]);
    }
  }

  void FeatureExtract(const UserFeatures& user, const std::vector<ItemFeatures>& item_list, const ContextFeatures& ctx, size_t batch_size,
                      std::vector<CrossFeatures>& cross_list, std::vector<BehaviorFeatures> behavior_list) {
    if (item_list.empty()) {
      return;
    }
    cross_list.resize(item_list.size());
    behavior_list.resize(item_list.size());

    // Calculate number of batches
    size_t num_batches = (item_list.size() + batch_size - 1) / batch_size;
    std::vector<std::thread> threads;
    threads.reserve(num_batches);

    // Feature extraction in parallel manner.
    for (size_t batch = 0; batch < num_batches; ++batch) {
      size_t start = batch * batch_size;
      size_t end = std::min(start + batch_size, item_list.size());

      threads.emplace_back([this, &user, &item_list, &ctx, &cross_list, &behavior_list, start, end]() {
        for (size_t i = start; i < end; ++i) {
          ComputeCrossFeatures(user, item_list[i], ctx, cross_list[i]);
          ComputeBehaviorFeatures(user, item_list[i], behavior_list[i]);
        }
      });
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
      thread.join();
    }
  }

 private:
  void ComputeCrossFeatures(const UserFeatures& user, const ItemFeatures& item, const ContextFeatures& ctx,
                            CrossFeatures& crossess) {
    ComputeUserItemCrossFeatures(user, item, crossess);
    ComputeContextItemCrossFeatures(item, ctx, crossess);
  }

  void ComputeUserItemCrossFeatures(const UserFeatures& user, const ItemFeatures& item,
                                    CrossFeatures& crossess) {
    OP_cal_topic_affinity(user, item, crossess);
    OP_cal_publisher_preference(user, item, crossess);
  }

  void ComputeContextItemCrossFeatures(const ItemFeatures& item, const ContextFeatures& ctx,
                                       CrossFeatures& crossess) {
  }

  void ComputeBehaviorFeatures(const UserFeatures& user, const ItemFeatures& item, BehaviorFeatures& behaviors) {
    // Hard search.
    // Hard-search means select the behavior data belongs to the same category of the candidate item.
    OP_cal_cat1_in_ubt(user, item, behaviors);
    OP_cal_cat2_in_ubt(user, item, behaviors);
    OP_cal_tags_in_ubt(user, item, behaviors);
  }

 private:
  void OP_cal_topic_affinity(const UserFeatures& user, const ItemFeatures& item,
                             CrossFeatures& crossess) {
    auto it = user.stats.topic_click_rates.find(item.topic);
    if (it != user.stats.topic_click_rates.end()) {
      crossess.topic_affinity = it->second;
      return;
    }
    crossess.topic_affinity = 0.0f;
  }

  void OP_cal_publisher_preference(const UserFeatures& user, const ItemFeatures& item,
                                   CrossFeatures& crossess) {
    auto it = user.stats.publisher_dwell_time.find(item.publisher);
    if (it != user.stats.publisher_dwell_time.end()) {
      crossess.publisher_preference = it->second;
      return;
    }
    crossess.publisher_preference = 0.0f;
  }

  void OP_cal_cat1_in_ubt(const UserFeatures& user, const ItemFeatures& item, BehaviorFeatures& behaviors) {
    auto it = user.ubt.cat_behavior_.find(item.cat1);
    if (it != user.ubt.cat_behavior_.end()) {
      for (const auto& item_features : it->second) {
        behaviors.cat1_list.emplace_back(item_features.cat1);
      }
    }
  }

  void OP_cal_cat2_in_ubt(const UserFeatures& user, const ItemFeatures& item, BehaviorFeatures& behaviors) {
    auto it = user.ubt.cat_behavior_.find(item.cat1);
    if (it != user.ubt.cat_behavior_.end()) {
      for (const auto& item_features : it->second) {
        behaviors.cat2_list.emplace_back(item_features.cat2);
      }
    }
  }

  void OP_cal_tags_in_ubt(const UserFeatures& user, const ItemFeatures& item, BehaviorFeatures& behaviors) {
    auto it = user.ubt.cat_behavior_.find(item.cat1);
    if (it != user.ubt.cat_behavior_.end()) {
      for (const auto& item_features : it->second) {
        behaviors.tag_list.emplace_back(item_features.tags);
      }
    }
  }

};

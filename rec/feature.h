#pragma once

#include <string>
#include <unordered_map>

struct ItemFeatures;

struct UserFeatures {
  int uid;
  int age;
  std::string gender;

  struct StatFeature {
    std::unordered_map<std::string, float> topic_click_rates;     // e.g., {"politics": 0.15, "sports": 0.08}
    std::unordered_map<std::string, float> publisher_dwell_time;  // e.g., {"bbc": 45.2, "cnn": 32.1}
  };
  StatFeature stats;

  struct UBT {
    // According to the sim paper,
    // Hard-search means select the behavior data belongs to the same category of the candidate item.
    // It's actually a inverted index.
    std::unordered_map<int, std::vector<ItemFeatures>> cat_behavior_;
  };
  UBT ubt;
};

struct ItemFeatures {
  int news_id;
  std::string topic;
  std::string publisher;
  time_t  publish_time;

  int cat1;
  int cat2;
  std::vector<int> tags;
};

struct ContextFeatures {
  std::string city;             // Derived from ip.
  std::string connection_type;  // "wifi", "4g"
};

// A property of the relationship between user, item and context.
// Lifecycle and Timing: User and item features are often precomputed offline in batch jobs (e.g., a daily job to calculate a user's topic affinity).
// Cross features, however, are often computed on-the-fly during request serving.
// For a given user, you retrieve their precomputed UserFeatures and for each candidate item, you retrieve its ItemFeatures.
// You then join them in real-time to create the CrossFeatures for that specific pair. Their lifecycle is inherently different.
//
// Cross Features are explicit interactions while the dnn model learns the implicit interactions between user and items.
struct alignas(64) CrossFeatures {
  // False sharing occurs when multiple threads frequently write to different memory locations that happen to reside on the same cache line
  // (typically 64 bytes on modern CPUs).
  // Even though the threads are accessing different array elements, if those elements are close enough in memory,
  // they might share the same cache line.
  // Use alignas(64) to ensure each CrossFeatures occupies a full cache line

  // 1.User-Item crosses.
  float topic_affinity = 0.0f;
  float publisher_preference = 0.0f;

  // 2.User-Context Crosses.

  // 3.Item-Context Crosses.

  // 4.High-Order Crosses.(User-Item-Context)

  // 5.Hashed Categorical Features ()

  // any other computed cross features.
};

struct alignas(64) BehaviorFeatures {
  // We also call these features side info since long time sequence can not be retrieved in index service.
  std::vector<int> cat1_list;
  std::vector<int> cat2_list;
  std::vector<std::vector<int>> tag_list;
};

#include "feature_extractor.h"

void Ranking(const UserFeatures& user_features, const std::vector<ItemFeatures>& items_features, const ContextFeatures& ctx_features) {
  std::vector<CrossFeatures> crossess_features;
  std::vector<BehaviorFeatures> behaviors_features;

  // cross_features and behavior_features are computed real-time.
  FeatureExtractor fe;
  fe.FeatureExtract(user_features, items_features, ctx_features, crossess_features, behaviors_features);

  for (int i = 0; i < items_features.size(); ++i) {
    // ModelInput model_input;
    // model_input.add(user_features);
    // model_input.add(items_features[i]);
    // model_input.add(ctx_features);

    // model_input.add(cross_features[i]);
    // model_input.add(behavior_features[i]);

    // auto score = model->predict(model_input);
  }

}

int main(void) {
  return 0;
}

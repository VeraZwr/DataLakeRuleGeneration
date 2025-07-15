import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict
import math


class ClusterBasedColumnMatcher:
    """
    Column matcher that leverages existing clustering results to generate
    refined rules and handle within-cluster and cross-cluster matching.
    """

    def __init__(self):
        # Cluster-aware thresholds
        self.thresholds = {
            'within_cluster_high': 0.85,  # High confidence within same cluster
            'within_cluster_medium': 0.70,  # Medium confidence within cluster
            'cross_cluster_possible': 0.60,  # Possible match across clusters
            'refinement_threshold': 0.55,  # Threshold for cluster refinement
            'merge_threshold': 0.75  # Threshold for cluster merging
        }

        # Cluster-specific feature importance
        self.cluster_feature_weights = {}

        # Learned cluster patterns
        self.cluster_profiles = {}

        # Error patterns within clusters
        self.error_patterns = {}

    def analyze_clusters(self, clustered_features: Dict[int, List[Dict]]) -> Dict:
        """
        Analyze existing clusters to learn patterns and generate cluster-specific rules.

        Args:
            clustered_features: {cluster_id: [feature_dict1, feature_dict2, ...]}

        Returns:
            Analysis results with cluster profiles and rules
        """
        cluster_analysis = {}

        for cluster_id, features_list in clustered_features.items():
            if len(features_list) < 2:
                continue

            # Analyze cluster characteristics
            profile = self._analyze_cluster_profile(features_list)

            # Generate cluster-specific rules
            rules = self._generate_cluster_rules(features_list, profile)

            # Detect error patterns within cluster
            error_patterns = self._detect_error_patterns(features_list)

            cluster_analysis[cluster_id] = {
                'profile': profile,
                'rules': rules,
                'error_patterns': error_patterns,
                'feature_importance': self._calculate_feature_importance(features_list),
                'quality_metrics': self._calculate_cluster_quality(features_list)
            }

        # Store for future use
        self.cluster_profiles = {cid: analysis['profile']
                                 for cid, analysis in cluster_analysis.items()}

        return cluster_analysis

    def match_within_cluster(self, features1: Dict, features2: Dict,
                             cluster_id: int) -> Tuple[float, List[str]]:
        """
        Match two columns within the same cluster using cluster-specific rules.
        """
        if cluster_id not in self.cluster_profiles:
            return self._fallback_match(features1, features2)

        profile = self.cluster_profiles[cluster_id]

        # Use cluster-specific matching strategy
        if profile['type'] == 'numeric':
            return self._numeric_cluster_match(features1, features2, profile)
        elif profile['type'] == 'text':
            return self._text_cluster_match(features1, features2, profile)
        elif profile['type'] == 'structured':
            return self._structured_cluster_match(features1, features2, profile)
        elif profile['type'] == 'mixed':
            return self._mixed_cluster_match(features1, features2, profile)
        else:
            return self._generic_cluster_match(features1, features2, profile)

    def match_across_clusters(self, features1: Dict, features2: Dict,
                              cluster1_id: int, cluster2_id: int) -> Tuple[float, List[str]]:
        """
        Match columns from different clusters - should be more restrictive.
        """
        # Check if clusters are compatible
        if not self._clusters_compatible(cluster1_id, cluster2_id):
            return 0.0, ["Clusters are incompatible"]

        # Use stricter cross-cluster matching
        similarities = self._calculate_cross_cluster_similarities(features1, features2)

        # Apply cross-cluster penalties
        base_score = np.mean(list(similarities.values()))
        cross_cluster_penalty = 0.15  # Reduce score for cross-cluster matches

        adjusted_score = max(0.0, base_score - cross_cluster_penalty)

        evidence = self._generate_cross_cluster_evidence(similarities, adjusted_score)

        return adjusted_score, evidence

    def refine_cluster_assignment(self, features: Dict, current_cluster: int,
                                  all_clusters: Dict[int, List[Dict]]) -> Tuple[int, float]:
        """
        Suggest better cluster assignment based on similarity rules.
        """
        best_cluster = current_cluster
        best_score = 0.0

        for cluster_id, cluster_features in all_clusters.items():
            if not cluster_features:
                continue

            # Calculate average similarity to cluster
            similarities = []
            for cluster_feature in cluster_features:
                if cluster_id == current_cluster:
                    sim, _ = self.match_within_cluster(features, cluster_feature, cluster_id)
                else:
                    sim, _ = self.match_across_clusters(features, cluster_feature,
                                                        current_cluster, cluster_id)
                similarities.append(sim)

            avg_similarity = np.mean(similarities) if similarities else 0.0

            if avg_similarity > best_score:
                best_score = avg_similarity
                best_cluster = cluster_id

        return best_cluster, best_score

    def detect_outliers_in_cluster(self, cluster_features: List[Dict],
                                   cluster_id: int) -> List[Tuple[int, float, List[str]]]:
        """
        Detect columns that might be outliers within their cluster.
        """
        outliers = []

        if len(cluster_features) < 3:
            return outliers

        for i, target_feature in enumerate(cluster_features):
            similarities = []

            # Compare with all other features in cluster
            for j, other_feature in enumerate(cluster_features):
                if i != j:
                    sim, _ = self.match_within_cluster(target_feature, other_feature, cluster_id)
                    similarities.append(sim)

            avg_similarity = np.mean(similarities)

            # Flag as outlier if significantly different from cluster
            if avg_similarity < self.thresholds['refinement_threshold']:
                evidence = [
                    f"Low average similarity to cluster: {avg_similarity:.3f}",
                    f"Significantly different from {len(similarities)} other columns"
                ]
                outliers.append((i, avg_similarity, evidence))

        return sorted(outliers, key=lambda x: x[1])  # Sort by similarity (lowest first)

    def suggest_cluster_merges(self, cluster_analysis: Dict) -> List[Tuple[int, int, float]]:
        """
        Suggest clusters that might benefit from merging.
        """
        merge_suggestions = []
        cluster_ids = list(cluster_analysis.keys())

        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                cluster1_id = cluster_ids[i]
                cluster2_id = cluster_ids[j]

                # Calculate inter-cluster similarity
                profile1 = cluster_analysis[cluster1_id]['profile']
                profile2 = cluster_analysis[cluster2_id]['profile']

                similarity = self._calculate_profile_similarity(profile1, profile2)

                if similarity > self.thresholds['merge_threshold']:
                    merge_suggestions.append((cluster1_id, cluster2_id, similarity))

        return sorted(merge_suggestions, key=lambda x: x[2], reverse=True)

    def _analyze_cluster_profile(self, features_list: List[Dict]) -> Dict:
        """Analyze the common characteristics of a cluster."""
        profile = {
            'size': len(features_list),
            'type': 'unknown',
            'common_patterns': [],
            'typical_ranges': {},
            'dominant_features': {},
            'variance_features': {}
        }

        # Determine cluster type
        data_types = [f.get('basic_data_type', 'unknown') for f in features_list]
        numeric_ratio = sum(1 for dt in data_types if dt in ['integer', 'floating']) / len(data_types)

        if numeric_ratio > 0.8:
            profile['type'] = 'numeric'
        elif all(f.get('dominant_pattern') for f in features_list):
            profile['type'] = 'structured'
        elif np.mean([f.get('characters_alphabet', 0) for f in features_list]) > 0.7:
            profile['type'] = 'text'
        else:
            profile['type'] = 'mixed'

        # Calculate typical ranges for numeric features
        numeric_features = ['null_ratio', 'unique_ratio', 'characters_numeric',
                            'characters_alphabet', 'avg_len', 'most_freq_value_ratio']

        for feature in numeric_features:
            values = [f.get(feature, 0) for f in features_list if f.get(feature) is not None]
            if values:
                profile['typical_ranges'][feature] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }

        # Find common patterns
        patterns = [f.get('dominant_pattern', '') for f in features_list if f.get('dominant_pattern')]
        if patterns:
            pattern_counts = defaultdict(int)
            for pattern in patterns:
                pattern_counts[pattern] += 1
            profile['common_patterns'] = sorted(pattern_counts.items(),
                                                key=lambda x: x[1], reverse=True)[:5]

        return profile

    def _generate_cluster_rules(self, features_list: List[Dict], profile: Dict) -> List[Dict]:
        """Generate rules specific to this cluster."""
        rules = []

        if profile['type'] == 'numeric':
            rules.extend(self._generate_numeric_rules(features_list, profile))
        elif profile['type'] == 'text':
            rules.extend(self._generate_text_rules(features_list, profile))
        elif profile['type'] == 'structured':
            rules.extend(self._generate_structured_rules(features_list, profile))

        # Add general cluster rules
        rules.extend(self._generate_general_cluster_rules(features_list, profile))

        return rules

    def _generate_numeric_rules(self, features_list: List[Dict], profile: Dict) -> List[Dict]:
        """Generate rules for numeric clusters."""
        rules = []

        # Range-based rules
        if 'numeric_min' in profile['typical_ranges']:
            range_info = profile['typical_ranges']['numeric_min']
            rules.append({
                'type': 'numeric_range',
                'feature': 'numeric_min',
                'tolerance': range_info['std'] * 2,
                'weight': 0.3
            })

        # Distribution-based rules
        if 'unique_ratio' in profile['typical_ranges']:
            unique_info = profile['typical_ranges']['unique_ratio']
            rules.append({
                'type': 'distribution_pattern',
                'feature': 'unique_ratio',
                'expected_range': (unique_info['min'], unique_info['max']),
                'weight': 0.25
            })

        return rules

    def _generate_text_rules(self, features_list: List[Dict], profile: Dict) -> List[Dict]:
        """Generate rules for text clusters."""
        rules = []

        # Keyword-based rules
        all_keywords = set()
        for features in features_list:
            if 'top_keywords' in features:
                all_keywords.update(features['top_keywords'].keys())

        if all_keywords:
            rules.append({
                'type': 'keyword_overlap',
                'keywords': all_keywords,
                'min_overlap': 0.2,
                'weight': 0.4
            })

        # Length-based rules
        if 'avg_len' in profile['typical_ranges']:
            len_info = profile['typical_ranges']['avg_len']
            rules.append({
                'type': 'length_pattern',
                'expected_range': (len_info['mean'] - len_info['std'],
                                   len_info['mean'] + len_info['std']),
                'weight': 0.2
            })

        return rules

    def _generate_structured_rules(self, features_list: List[Dict], profile: Dict) -> List[Dict]:
        """Generate rules for structured data clusters."""
        rules = []

        # Pattern-based rules
        if profile['common_patterns']:
            main_pattern = profile['common_patterns'][0][0]
            rules.append({
                'type': 'pattern_match',
                'pattern': main_pattern,
                'flexibility': 0.2,
                'weight': 0.5
            })

        return rules

    def _generate_general_cluster_rules(self, features_list: List[Dict], profile: Dict) -> List[Dict]:
        """Generate general rules applicable to any cluster."""
        rules = []

        # Null ratio consistency
        if 'null_ratio' in profile['typical_ranges']:
            null_info = profile['typical_ranges']['null_ratio']
            rules.append({
                'type': 'null_consistency',
                'expected_range': (null_info['min'], null_info['max']),
                'weight': 0.1
            })

        return rules

    def _detect_error_patterns(self, features_list: List[Dict]) -> Dict:
        """Detect common error patterns within a cluster."""
        error_patterns = {
            'high_null_columns': [],
            'low_uniqueness': [],
            'outlier_lengths': [],
            'inconsistent_types': []
        }

        for i, features in enumerate(features_list):
            # High null ratio
            if features.get('null_ratio', 0) > 0.5:
                error_patterns['high_null_columns'].append(i)

            # Low uniqueness (potential data quality issue)
            if features.get('unique_ratio', 1) < 0.1:
                error_patterns['low_uniqueness'].append(i)

            # Outlier lengths
            avg_len = features.get('avg_len', 0)
            if avg_len < 2 or avg_len > 100:
                error_patterns['outlier_lengths'].append(i)

        return error_patterns

    def _calculate_feature_importance(self, features_list: List[Dict]) -> Dict[str, float]:
        """Calculate which features are most important for this cluster."""
        importance = {}

        # Calculate variance for each feature
        numeric_features = ['null_ratio', 'unique_ratio', 'characters_numeric',
                            'characters_alphabet', 'avg_len']

        for feature in numeric_features:
            values = [f.get(feature, 0) for f in features_list]
            if values:
                variance = np.var(values)
                # Higher variance = more discriminative = more important
                importance[feature] = variance

        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    def _calculate_cluster_quality(self, features_list: List[Dict]) -> Dict:
        """Calculate quality metrics for the cluster."""
        if len(features_list) < 2:
            return {'cohesion': 0.0, 'consistency': 0.0}

        # Calculate pairwise similarities within cluster
        similarities = []
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                sim, _ = self._fallback_match(features_list[i], features_list[j])
                similarities.append(sim)

        cohesion = np.mean(similarities) if similarities else 0.0
        consistency = 1.0 - np.std(similarities) if similarities else 0.0

        return {
            'cohesion': cohesion,
            'consistency': max(0.0, consistency),
            'size': len(features_list)
        }

    def _numeric_cluster_match(self, f1: Dict, f2: Dict, profile: Dict) -> Tuple[float, List[str]]:
        """Specialized matching for numeric clusters."""
        score = 0.0
        evidence = []

        # Check numeric ranges
        if all(k in f1 and k in f2 for k in ['numeric_min', 'numeric_max']):
            range1 = f1['numeric_max'] - f1['numeric_min']
            range2 = f2['numeric_max'] - f2['numeric_min']

            if range1 > 0 and range2 > 0:
                range_sim = 1.0 - abs(range1 - range2) / max(range1, range2)
                score += range_sim * 0.4
                evidence.append(f"Numeric range similarity: {range_sim:.3f}")

        # Check distribution patterns
        if 'unique_ratio' in f1 and 'unique_ratio' in f2:
            unique_sim = 1.0 - abs(f1['unique_ratio'] - f2['unique_ratio'])
            score += unique_sim * 0.3
            evidence.append(f"Uniqueness similarity: {unique_sim:.3f}")

        # Check statistical quartiles
        quartile_features = ['Q1', 'Q2', 'Q3']
        if all(f in f1 and f in f2 for f in quartile_features):
            quartile_sims = []
            for qf in quartile_features:
                if f1[qf] is not None and f2[qf] is not None:
                    val1, val2 = f1[qf], f2[qf]
                    if val1 == val2:
                        quartile_sims.append(1.0)
                    else:
                        max_val = max(abs(val1), abs(val2))
                        if max_val > 0:
                            quartile_sims.append(1.0 - abs(val1 - val2) / max_val)

            if quartile_sims:
                quartile_sim = np.mean(quartile_sims)
                score += quartile_sim * 0.3
                evidence.append(f"Quartile similarity: {quartile_sim:.3f}")

        return min(1.0, score), evidence

    def _text_cluster_match(self, f1: Dict, f2: Dict, profile: Dict) -> Tuple[float, List[str]]:
        """Specialized matching for text clusters."""
        score = 0.0
        evidence = []

        # Keyword overlap
        if 'top_keywords' in f1 and 'top_keywords' in f2:
            keywords1 = set(f1['top_keywords'].keys())
            keywords2 = set(f2['top_keywords'].keys())

            if keywords1 or keywords2:
                intersection = len(keywords1 & keywords2)
                union = len(keywords1 | keywords2)
                jaccard = intersection / union if union > 0 else 0

                score += jaccard * 0.4
                evidence.append(f"Keyword overlap: {jaccard:.3f}")

        # Character composition
        char_features = ['characters_alphabet', 'characters_numeric', 'characters_punctuation']
        char_similarities = []

        for feature in char_features:
            if feature in f1 and feature in f2:
                val1, val2 = f1[feature], f2[feature]
                char_similarities.append(1.0 - abs(val1 - val2))

        if char_similarities:
            char_sim = np.mean(char_similarities)
            score += char_sim * 0.3
            evidence.append(f"Character composition similarity: {char_sim:.3f}")

        # Length similarity
        if 'avg_len' in f1 and 'avg_len' in f2:
            len1, len2 = f1['avg_len'], f2['avg_len']
            if len1 > 0 and len2 > 0:
                len_sim = 1.0 - abs(len1 - len2) / max(len1, len2)
                score += len_sim * 0.3
                evidence.append(f"Length similarity: {len_sim:.3f}")

        return min(1.0, score), evidence

    def _structured_cluster_match(self, f1: Dict, f2: Dict, profile: Dict) -> Tuple[float, List[str]]:
        """Specialized matching for structured data clusters."""
        score = 0.0
        evidence = []

        # Pattern matching
        pattern1 = f1.get('dominant_pattern', '')
        pattern2 = f2.get('dominant_pattern', '')

        if pattern1 and pattern2:
            if pattern1 == pattern2:
                pattern_sim = 1.0
            else:
                # Flexible pattern matching
                pattern_sim = self._flexible_pattern_similarity(pattern1, pattern2)

            score += pattern_sim * 0.6
            evidence.append(f"Pattern similarity: {pattern_sim:.3f}")

        # Length consistency
        length_features = ['max_len', 'min_len', 'avg_len']
        length_similarities = []

        for feature in length_features:
            if feature in f1 and feature in f2:
                val1, val2 = f1[feature], f2[feature]
                if val1 == 0 and val2 == 0:
                    length_similarities.append(1.0)
                elif val1 == 0 or val2 == 0:
                    length_similarities.append(0.0)
                else:
                    length_similarities.append(1.0 - abs(val1 - val2) / max(val1, val2))

        if length_similarities:
            length_sim = np.mean(length_similarities)
            score += length_sim * 0.4
            evidence.append(f"Length consistency: {length_sim:.3f}")

        return min(1.0, score), evidence

    def _mixed_cluster_match(self, f1: Dict, f2: Dict, profile: Dict) -> Tuple[float, List[str]]:
        """Specialized matching for mixed data clusters."""
        # Use a combination of all approaches with balanced weights
        numeric_score, numeric_evidence = self._numeric_cluster_match(f1, f2, profile)
        text_score, text_evidence = self._text_cluster_match(f1, f2, profile)

        # Weighted combination
        combined_score = (numeric_score * 0.5 + text_score * 0.5)
        combined_evidence = numeric_evidence + text_evidence

        return combined_score, combined_evidence

    def _generic_cluster_match(self, f1: Dict, f2: Dict, profile: Dict) -> Tuple[float, List[str]]:
        """Generic matching when cluster type is unknown."""
        return self._fallback_match(f1, f2)

    def _fallback_match(self, f1: Dict, f2: Dict) -> Tuple[float, List[str]]:
        """Fallback matching method."""
        score = 0.0
        evidence = []

        # Basic data type match
        if f1.get('basic_data_type') == f2.get('basic_data_type'):
            score += 0.3
            evidence.append("Data types match")

        # Null ratio similarity
        if 'null_ratio' in f1 and 'null_ratio' in f2:
            null_sim = 1.0 - abs(f1['null_ratio'] - f2['null_ratio'])
            score += null_sim * 0.3
            evidence.append(f"Null ratio similarity: {null_sim:.3f}")

        # Unique ratio similarity
        if 'unique_ratio' in f1 and 'unique_ratio' in f2:
            unique_sim = 1.0 - abs(f1['unique_ratio'] - f2['unique_ratio'])
            score += unique_sim * 0.4
            evidence.append(f"Uniqueness similarity: {unique_sim:.3f}")

        return min(1.0, score), evidence

    def _clusters_compatible(self, cluster1_id: int, cluster2_id: int) -> bool:
        """Check if two clusters are compatible for cross-cluster matching."""
        if cluster1_id not in self.cluster_profiles or cluster2_id not in self.cluster_profiles:
            return True  # Allow if we don't have enough info

        profile1 = self.cluster_profiles[cluster1_id]
        profile2 = self.cluster_profiles[cluster2_id]

        # Same type clusters are more compatible
        if profile1['type'] == profile2['type']:
            return True

        # Some types are compatible
        compatible_pairs = [
            ('numeric', 'mixed'),
            ('text', 'mixed'),
            ('structured', 'mixed')
        ]

        type_pair = (profile1['type'], profile2['type'])
        return type_pair in compatible_pairs or type_pair[::-1] in compatible_pairs

    def _calculate_cross_cluster_similarities(self, f1: Dict, f2: Dict) -> Dict[str, float]:
        """Calculate similarities for cross-cluster matching."""
        similarities = {}

        # More restrictive cross-cluster rules
        if f1.get('basic_data_type') == f2.get('basic_data_type'):
            similarities['data_type'] = 1.0
        else:
            similarities['data_type'] = 0.0

        # Stricter null ratio matching
        if 'null_ratio' in f1 and 'null_ratio' in f2:
            null_diff = abs(f1['null_ratio'] - f2['null_ratio'])
            similarities['null_ratio'] = max(0.0, 1.0 - null_diff * 2)  # More penalty

        # Stricter uniqueness matching
        if 'unique_ratio' in f1 and 'unique_ratio' in f2:
            unique_diff = abs(f1['unique_ratio'] - f2['unique_ratio'])
            similarities['unique_ratio'] = max(0.0, 1.0 - unique_diff * 1.5)  # More penalty

        return similarities

    def _generate_cross_cluster_evidence(self, similarities: Dict[str, float],
                                         score: float) -> List[str]:
        """Generate evidence for cross-cluster matches."""
        evidence = []

        if score > 0.6:
            evidence.append("CROSS-CLUSTER: Strong evidence for similarity")
        elif score > 0.4:
            evidence.append("CROSS-CLUSTER: Moderate evidence for similarity")
        else:
            evidence.append("CROSS-CLUSTER: Weak evidence for similarity")

        for feature, sim in similarities.items():
            if sim > 0.7:
                evidence.append(f"Strong {feature} match: {sim:.3f}")
            elif sim > 0.5:
                evidence.append(f"Moderate {feature} match: {sim:.3f}")

        return evidence

    def _calculate_profile_similarity(self, profile1: Dict, profile2: Dict) -> float:
        """Calculate similarity between two cluster profiles."""
        score = 0.0

        # Type similarity
        if profile1['type'] == profile2['type']:
            score += 0.4

        # Size similarity (similar cluster sizes might indicate similar data)
        size1, size2 = profile1['size'], profile2['size']
        if size1 > 0 and size2 > 0:
            size_sim = 1.0 - abs(size1 - size2) / max(size1, size2)
            score += size_sim * 0.2

        # Typical ranges overlap
        ranges1 = profile1.get('typical_ranges', {})
        ranges2 = profile2.get('typical_ranges', {})

        range_similarities = []
        for feature in set(ranges1.keys()) & set(ranges2.keys()):
            r1, r2 = ranges1[feature], ranges2[feature]
            mean_diff = abs(r1['mean'] - r2['mean'])
            max_mean = max(abs(r1['mean']), abs(r2['mean']))

            if max_mean > 0:
                range_sim = 1.0 - mean_diff / max_mean
                range_similarities.append(range_sim)

        if range_similarities:
            score += np.mean(range_similarities) * 0.4

        return min(1.0, score)

    def _flexible_pattern_similarity(self, pattern1: str, pattern2: str) -> float:
        """Calculate flexible pattern similarity."""
        if not pattern1 or not pattern2:
            return 0.5

        # Exact match
        if pattern1 == pattern2:
            return 1.0

        # Length similarity
        len_sim = 1.0 - abs(len(pattern1) - len(pattern2)) / max(len(pattern1), len(pattern2))

        # Character type similarity
        def char_types(pattern):
            return {
                'digits': len([c for c in pattern if c.isdigit()]),
                'letters': len([c for c in pattern if c.isalpha()]),
                'special': len([c for c in pattern if not c.isalnum()])
            }

        types1 = char_types(pattern1)
        types2 = char_types(pattern2)

        type_similarities = []
        for key in types1:
            if types1[key] == 0 and types2[key] == 0:
                type_similarities.append(1.0)
            elif types1[key] == 0 or types2[key] == 0:
                type_similarities.append(0.0)
            else:
                type_similarities.append(1.0 - abs(types1[key] - types2[key]) / max(types1[key], types2[key]))

        type_sim = np.mean(type_similarities)

        return (len_sim * 0.3 + type_sim * 0.7)


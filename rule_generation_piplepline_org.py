import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RuleLevel(Enum):
    LEVEL_1 = 1  # Universal rules
    LEVEL_2 = 2  # Domain-specific rules
    LEVEL_3 = 3  # Table-specific rules
    LEVEL_4 = 4  # Instance-specific rules


@dataclass
class DataQualityRule:
    id: str
    name: str
    level: RuleLevel
    condition: str
    parameters: Dict
    domain: Optional[str] = None
    success_rate: float = 1.0
    confidence: float = 1.0
    created_from: Optional[str] = None  # Source table/domain


class SimilarityScores:
    def __init__(self, schema: float, pattern: float, domain: float):
        self.schema = schema
        self.pattern = pattern
        self.domain = domain
        self.overall = (schema + pattern + domain) / 3


@dataclass
class TableSchema:
    name: str
    columns: List[str]
    data_types: Dict[str, str]
    domain: str
    patterns: Dict[str, List[str]] = field(default_factory=dict)
    statistics: Dict = field(default_factory=dict)


class RuleAdaptationMechanisms:
    @staticmethod
    def parameterize_rule(rule: DataQualityRule, target_stats: Dict) -> DataQualityRule:
        """Adjust numerical thresholds based on target table statistics"""
        adapted_rule = DataQualityRule(
            id=f"{rule.id}_adapted",
            name=rule.name,
            level=rule.level,
            condition=rule.condition,
            parameters=rule.parameters.copy(),
            domain=rule.domain,
            success_rate=rule.success_rate,
            confidence=rule.confidence * 0.9,  # Reduce confidence for adapted rules
            created_from=rule.id
        )

        # Adapt numerical thresholds
        if 'threshold' in rule.parameters:
            if 'mean' in target_stats:
                # Adjust threshold based on target data distribution
                adapted_rule.parameters['threshold'] = target_stats['mean'] * rule.parameters.get(
                    'threshold_multiplier', 1.0)

        if 'min_length' in rule.parameters and 'avg_length' in target_stats:
            adapted_rule.parameters['min_length'] = max(1, int(target_stats['avg_length'] * 0.1))

        return adapted_rule

    @staticmethod
    def generalize_pattern(pattern: str) -> str:
        """Abstract specific patterns to more general forms"""
        # Replace specific numbers with wildcards
        pattern = re.sub(r'\d+', r'\\d+', pattern)
        # Replace specific strings with character classes
        pattern = re.sub(r'[A-Z]{2,}', r'[A-Z]+', pattern)
        pattern = re.sub(r'[a-z]{2,}', r'[a-z]+', pattern)
        return pattern

    @staticmethod
    def map_domain_context(rule: DataQualityRule, source_domain: str, target_domain: str) -> DataQualityRule:
        """Map domain-specific terms to equivalent concepts in target domain"""
        domain_mappings = {
            'finance': {'account': 'customer', 'transaction': 'record'},
            'healthcare': {'patient': 'customer', 'diagnosis': 'category'},
            'retail': {'customer': 'client', 'product': 'item'}
        }

        mapped_rule = DataQualityRule(
            id=f"{rule.id}_mapped",
            name=rule.name,
            level=rule.level,
            condition=rule.condition,
            parameters=rule.parameters.copy(),
            domain=target_domain,
            success_rate=rule.success_rate,
            confidence=rule.confidence * 0.8,  # Reduce confidence for mapped rules
            created_from=rule.id
        )

        # Apply domain mappings to rule condition
        if source_domain in domain_mappings and target_domain in domain_mappings:
            for source_term, target_term in domain_mappings[source_domain].items():
                mapped_rule.condition = mapped_rule.condition.replace(source_term, target_term)

        return mapped_rule


class SimilarityCalculator:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def calculate_schema_similarity(self, source: TableSchema, target: TableSchema) -> float:
        """Calculate schema similarity based on column names and types"""
        source_cols = set(source.columns)
        target_cols = set(target.columns)

        # Jaccard similarity for column names
        intersection = len(source_cols.intersection(target_cols))
        union = len(source_cols.union(target_cols))
        column_similarity = intersection / union if union > 0 else 0

        # Data type similarity
        common_cols = source_cols.intersection(target_cols)
        type_matches = sum(1 for col in common_cols
                           if source.data_types.get(col) == target.data_types.get(col))
        type_similarity = type_matches / len(common_cols) if common_cols else 0

        return (column_similarity + type_similarity) / 2

    def calculate_pattern_similarity(self, source: TableSchema, target: TableSchema) -> float:
        """Calculate pattern similarity using TF-IDF"""
        source_patterns = ' '.join([' '.join(patterns) for patterns in source.patterns.values()])
        target_patterns = ' '.join([' '.join(patterns) for patterns in target.patterns.values()])

        if not source_patterns or not target_patterns:
            return 0.0

        try:
            tfidf_matrix = self.vectorizer.fit_transform([source_patterns, target_patterns])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            return 0.0

    def calculate_domain_similarity(self, source: TableSchema, target: TableSchema) -> float:
        """Calculate domain similarity"""
        if source.domain == target.domain:
            return 1.0

        # Domain hierarchy mapping
        domain_hierarchy = {
            'finance': ['banking', 'insurance', 'investment'],
            'healthcare': ['medical', 'pharmacy', 'clinical'],
            'retail': ['ecommerce', 'sales', 'inventory']
        }

        for parent, children in domain_hierarchy.items():
            if (source.domain == parent and target.domain in children) or \
                    (target.domain == parent and source.domain in children) or \
                    (source.domain in children and target.domain in children):
                return 0.7

        return 0.0

    def calculate_overall_similarity(self, source: TableSchema, target: TableSchema) -> SimilarityScores:
        """Calculate overall similarity scores"""
        schema_sim = self.calculate_schema_similarity(source, target)
        pattern_sim = self.calculate_pattern_similarity(source, target)
        domain_sim = self.calculate_domain_similarity(source, target)

        return SimilarityScores(schema_sim, pattern_sim, domain_sim)


class TransferProcessFlow:
    def __init__(self):
        self.rules_repository: Dict[str, List[DataQualityRule]] = defaultdict(list)
        self.similarity_calculator = SimilarityCalculator()
        self.adaptation_mechanisms = RuleAdaptationMechanisms()
        self.transfer_history: List[Dict] = []
        self.similarity_thresholds = {
            RuleLevel.LEVEL_1: 0.0,  # Universal rules always apply
            RuleLevel.LEVEL_2: 0.5,  # Domain-specific rules
            RuleLevel.LEVEL_3: 0.7,  # Table-specific rules
            RuleLevel.LEVEL_4: 0.9  # Instance-specific rules
        }

    def add_rule(self, table_name: str, rule: DataQualityRule):
        """Add a rule to the repository"""
        self.rules_repository[table_name].append(rule)

    def apply_universal_rules(self, target_table: TableSchema) -> List[DataQualityRule]:
        """Apply Level 1 (universal) rules to all new tables automatically"""
        universal_rules = []

        for table_rules in self.rules_repository.values():
            for rule in table_rules:
                if rule.level == RuleLevel.LEVEL_1:
                    universal_rules.append(rule)

        return universal_rules

    def calculate_similarity_scores(self, source_table: TableSchema, target_table: TableSchema) -> SimilarityScores:
        """Calculate schema, pattern, and domain similarity scores"""
        return self.similarity_calculator.calculate_overall_similarity(source_table, target_table)

    def hierarchical_filtering(self, source_table: TableSchema, target_table: TableSchema) -> List[
        Tuple[DataQualityRule, float]]:
        """Progressively apply rules from Level 2 downward based on similarity thresholds"""
        similarity_scores = self.calculate_similarity_scores(source_table, target_table)
        filtered_rules = []

        source_rules = self.rules_repository.get(source_table.name, [])

        for rule in source_rules:
            if rule.level == RuleLevel.LEVEL_1:
                # Universal rules always apply
                filtered_rules.append((rule, 1.0))
            else:
                # Check if similarity meets threshold for this rule level
                threshold = self.similarity_thresholds.get(rule.level, 0.5)
                if similarity_scores.overall >= threshold:
                    # Calculate confidence based on similarity and rule level
                    confidence = self._calculate_transfer_confidence(rule, similarity_scores)
                    filtered_rules.append((rule, confidence))

        return filtered_rules

    def _calculate_transfer_confidence(self, rule: DataQualityRule, similarity: SimilarityScores) -> float:
        """Calculate confidence for rule transfer based on similarity and rule characteristics"""
        base_confidence = rule.confidence
        similarity_factor = similarity.overall
        level_penalty = {
            RuleLevel.LEVEL_1: 1.0,
            RuleLevel.LEVEL_2: 0.9,
            RuleLevel.LEVEL_3: 0.8,
            RuleLevel.LEVEL_4: 0.7
        }.get(rule.level, 0.5)

        success_rate_factor = rule.success_rate

        return base_confidence * similarity_factor * level_penalty * success_rate_factor

    def confidence_ranking(self, rules_with_confidence: List[Tuple[DataQualityRule, float]]) -> List[
        Tuple[DataQualityRule, float]]:
        """Rank transferred rules by confidence levels for error detection prioritization"""
        return sorted(rules_with_confidence, key=lambda x: x[1], reverse=True)

    def adapt_rules(self, rules_with_confidence: List[Tuple[DataQualityRule, float]],
                    target_table: TableSchema) -> List[DataQualityRule]:
        """Apply rule adaptation mechanisms"""
        adapted_rules = []

        for rule, confidence in rules_with_confidence:
            adapted_rule = rule

            # Apply parameterization
            if target_table.statistics:
                adapted_rule = self.adaptation_mechanisms.parameterize_rule(adapted_rule, target_table.statistics)

            # Apply pattern generalization for pattern-based rules
            if 'pattern' in rule.parameters:
                generalized_pattern = self.adaptation_mechanisms.generalize_pattern(rule.parameters['pattern'])
                adapted_rule.parameters['pattern'] = generalized_pattern

            # Apply context mapping if domains differ
            if rule.domain and rule.domain != target_table.domain:
                adapted_rule = self.adaptation_mechanisms.map_domain_context(
                    adapted_rule, rule.domain, target_table.domain
                )

            # Apply confidence weighting
            adapted_rule.confidence = confidence
            adapted_rules.append(adapted_rule)

        return adapted_rules

    def transfer_rules(self, source_table: TableSchema, target_table: TableSchema) -> List[DataQualityRule]:
        """Main transfer process implementation"""
        # Step 1: Apply universal rules
        universal_rules = self.apply_universal_rules(target_table)
        universal_rules_with_confidence = [(rule, 1.0) for rule in universal_rules]

        # Step 2: Calculate similarity and apply hierarchical filtering
        filtered_rules = self.hierarchical_filtering(source_table, target_table)

        # Combine universal and filtered rules
        all_rules_with_confidence = universal_rules_with_confidence + filtered_rules

        # Step 3: Rank by confidence
        ranked_rules = self.confidence_ranking(all_rules_with_confidence)

        # Step 4: Adapt rules
        adapted_rules = self.adapt_rules(ranked_rules, target_table)

        # Log transfer for adaptive learning
        self._log_transfer(source_table, target_table, adapted_rules)

        return adapted_rules

    def _log_transfer(self, source_table: TableSchema, target_table: TableSchema,
                      transferred_rules: List[DataQualityRule]):
        """Log transfer for learning purposes"""
        transfer_record = {
            'source_table': source_table.name,
            'target_table': target_table.name,
            'similarity_scores': self.calculate_similarity_scores(source_table, target_table).__dict__,
            'transferred_rules': [rule.id for rule in transferred_rules],
            'timestamp': np.datetime64('now')
        }
        self.transfer_history.append(transfer_record)

    def update_success_rates(self, validation_results: Dict[str, Dict[str, float]]):
        """Update transfer success rates based on validation results - Adaptive Learning"""
        for table_name, rule_results in validation_results.items():
            if table_name in self.rules_repository:
                for rule in self.rules_repository[table_name]:
                    if rule.id in rule_results:
                        # Update success rate using exponential moving average
                        alpha = 0.1  # Learning rate
                        new_success_rate = rule_results[rule.id]
                        rule.success_rate = (1 - alpha) * rule.success_rate + alpha * new_success_rate

    def get_transfer_statistics(self) -> Dict:
        """Get statistics about rule transfers"""
        if not self.transfer_history:
            return {}

        stats = {
            'total_transfers': len(self.transfer_history),
            'avg_similarity_scores': {
                'schema': np.mean([t['similarity_scores']['schema'] for t in self.transfer_history]),
                'pattern': np.mean([t['similarity_scores']['pattern'] for t in self.transfer_history]),
                'domain': np.mean([t['similarity_scores']['domain'] for t in self.transfer_history]),
                'overall': np.mean([t['similarity_scores']['overall'] for t in self.transfer_history])
            },
            'rules_per_transfer': np.mean([len(t['transferred_rules']) for t in self.transfer_history])
        }

        return stats


# Example usage and testing
def create_example_usage():
    # Initialize the transfer system
    transfer_system = TransferProcessFlow()

    # Create example schemas
    source_schema = TableSchema(
        name='customer_data',
        columns=['customer_id', 'email', 'phone', 'address'],
        data_types={'customer_id': 'int', 'email': 'string', 'phone': 'string', 'address': 'string'},
        domain='retail',
        patterns={'email': ['.*@.*\\..*'], 'phone': ['\\d{3}-\\d{3}-\\d{4}']},
        statistics={'mean': 1000, 'avg_length': 50}
    )

    target_schema = TableSchema(
        name='client_info',
        columns=['client_id', 'email', 'telephone', 'location'],
        data_types={'client_id': 'int', 'email': 'string', 'telephone': 'string', 'location': 'string'},
        domain='finance',
        patterns={'email': ['.*@.*\\..*'], 'telephone': ['\\(\\d{3}\\) \\d{3}-\\d{4}']},
        statistics={'mean': 800, 'avg_length': 45}
    )

    # Create example rules
    universal_rule = DataQualityRule(
        id='not_null_check',
        name='Not Null Validation',
        level=RuleLevel.LEVEL_1,
        condition='column IS NOT NULL',
        parameters={'columns': ['id']},
        confidence=1.0
    )

    domain_rule = DataQualityRule(
        id='email_format_check',
        name='Email Format Validation',
        level=RuleLevel.LEVEL_2,
        condition='email MATCHES pattern',
        parameters={'pattern': '.*@.*\\..*', 'column': 'email'},
        domain='retail',
        confidence=0.9
    )

    # Add rules to repository
    transfer_system.add_rule('customer_data', universal_rule)
    transfer_system.add_rule('customer_data', domain_rule)

    # Transfer rules
    transferred_rules = transfer_system.transfer_rules(source_schema, target_schema)

    # Print results
    print("Transfer Process Results:")
    print(f"Number of transferred rules: {len(transferred_rules)}")
    for rule in transferred_rules:
        print(f"- {rule.name} (Confidence: {rule.confidence:.2f})")

    # Simulate validation results and update success rates
    validation_results = {
        'customer_data': {
            'not_null_check': 0.95,
            'email_format_check': 0.88
        }
    }
    transfer_system.update_success_rates(validation_results)

    # Get statistics
    stats = transfer_system.get_transfer_statistics()
    print(f"\nTransfer Statistics: {stats}")


if __name__ == "__main__":
    create_example_usage()
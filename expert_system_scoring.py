"""
Expert System Scoring Module for Cr³⁺ Phosphor Candidate Selection
Author: Snežana Đurković
Year: 2026
Description: Evaluates ML predictions and provides synthesis recommendations
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import re


class PhosphorExpertSystem:
    """
    Ekspertski sistem za evaluaciju Cr³⁺ fosfora kandidata.
    Kombinuje ML predvikcije sa domenskim znanjem za optimalne preporuke.
    """

    def __init__(self,
                 target_dqb_range: Tuple[float, float] = (2.8, 3.8),
                 target_emission_range: Tuple[float, float] = (650, 720)):
        """
        Inicijalizacija ekspertskog sistema.

        Args:
            target_dqb_range: Ciljni opseg Dq/B vrednosti
            target_emission_range: Ciljni opseg emisione talasne dužine (nm)
        """
        self.target_dqb_range = target_dqb_range
        self.target_emission_range = target_emission_range

        # Definisanje teških metala i toksičnih elemenata
        self.toxic_elements = ['Cd', 'Pb', 'Hg', 'As', 'Tl']
        self.exotic_elements = ['Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au']
        self.rare_earth = ['La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb',
                           'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Y', 'Sc']

        # Precursor availability database
        self.precursor_availability = {
            'common': ['Ca', 'Sr', 'Ba', 'Mg', 'Al', 'Ga', 'Ti', 'Zr', 'Si', 'Ge', 'Zn'],
            'available': ['La', 'Y', 'Gd', 'Lu', 'Sc', 'Nb', 'Ta'],
            'expensive': ['In', 'W', 'Mo'],
            'toxic': self.toxic_elements
        }

    def dqb_to_emission_wavelength(self, dqb: float) -> float:
        """
        Konvertuje Dq/B u emisionu talasnu dužinu koristeći Tanabe-Sugano dijagram.

        Empirijska relacija za Cr³⁺:
        - Dq/B < 2.3: NIR emission (>750 nm) - ⁴T₂g lowest
        - Dq/B = 2.3-2.8: Red-NIR transition (~700-750 nm)
        - Dq/B = 2.8-3.8: Red emission (650-700 nm) - target range
        - Dq/B > 3.8: Deep red (620-650 nm), potentially unstable

        Args:
            dqb: Dq/B ratio

        Returns:
            emission_wavelength: Predicted emission wavelength in nm
        """
        if dqb < 2.0:
            return 800  # Deep NIR
        elif dqb < 2.3:
            return 750 - (dqb - 2.0) * 100  # 750-780 nm
        elif dqb < 2.8:
            return 700 - (dqb - 2.3) * 80  # 700-750 nm
        elif dqb < 3.8:
            return 660 - (dqb - 2.8) * 30  # 650-690 nm (optimal red)
        else:
            return 620 - (dqb - 3.8) * 20  # 600-620 nm (deep red)

    def calculate_performance_score(self, dqb_prediction: float) -> Dict[str, float]:
        """
        Ocenjuje performanse kandidata na osnovu Dq/B i emisione talasne dužine.

        Args:
            dqb_prediction: Predviđena Dq/B vrednost

        Returns:
            Dictionary sa performance metrikama
        """
        emission_wl = self.dqb_to_emission_wavelength(dqb_prediction)
        min_wl, max_wl = self.target_emission_range
        min_dqb, max_dqb = self.target_dqb_range

        # Emission match score
        if min_wl <= emission_wl <= max_wl:
            emission_score = 100
        elif emission_wl < min_wl:
            # Previše plava (deep red)
            emission_score = max(0, 100 - 2 * (min_wl - emission_wl))
        else:
            # Previše crvena (NIR)
            emission_score = max(0, 100 - 1.5 * (emission_wl - max_wl))

        # Dq/B match score
        if min_dqb <= dqb_prediction <= max_dqb:
            dqb_score = 100
        elif dqb_prediction < min_dqb:
            dqb_score = max(0, 100 - 25 * (min_dqb - dqb_prediction))
        else:
            dqb_score = max(0, 100 - 20 * (dqb_prediction - max_dqb))

        # Thermal stability estimate (higher Dq/B = better thermal stability)
        thermal_score = min(100, 50 + 15 * dqb_prediction)

        # Overall performance score
        performance_score = (0.5 * emission_score +
                             0.3 * dqb_score +
                             0.2 * thermal_score)

        return {
            'performance_score': performance_score,
            'emission_wavelength': emission_wl,
            'emission_score': emission_score,
            'dqb_score': dqb_score,
            'thermal_stability_score': thermal_score
        }

    def calculate_confidence_score(self,
                                   catboost_pred: float,
                                   nn_pred: float,
                                   uncertainty: float) -> Dict[str, float]:
        """
        Ocenjuje pouzdanost predviđanja.

        Args:
            catboost_pred: CatBoost predviđanje
            nn_pred: Neural Network predviđanje
            uncertainty: Standard deviation across CV folds

        Returns:
            Dictionary sa confidence metrikama
        """
        # Model agreement score
        model_disagreement = abs(catboost_pred - nn_pred)

        if model_disagreement < 0.1:
            agreement_score = 100
        elif model_disagreement < 0.2:
            agreement_score = 90
        elif model_disagreement < 0.3:
            agreement_score = 75
        elif model_disagreement < 0.5:
            agreement_score = 55
        else:
            agreement_score = 30

        # Uncertainty score (lower is better)
        if uncertainty < 0.05:
            uncertainty_score = 100
        elif uncertainty < 0.10:
            uncertainty_score = 90
        elif uncertainty < 0.15:
            uncertainty_score = 75
        elif uncertainty < 0.25:
            uncertainty_score = 55
        else:
            uncertainty_score = 30

        # Combined confidence score
        confidence_score = 0.55 * agreement_score + 0.45 * uncertainty_score

        return {
            'confidence_score': confidence_score,
            'model_agreement_score': agreement_score,
            'uncertainty_score': uncertainty_score,
            'model_disagreement': model_disagreement
        }

    def parse_formula_elements(self, formula: str) -> list:
        """
        Ekstraktuje sve elemente iz hemijske formule.

        Args:
            formula: Hemijska formula (npr. "Ca2MgWO6")

        Returns:
            Lista elemenata
        """
        # Regex za kapitalizovane elemente
        elements = re.findall(r'[A-Z][a-z]?', formula)
        return list(set(elements))  # Unique elements

    def calculate_feasibility_score(self, formula: str) -> Dict[str, float]:
        """
        Ocenjuje sintetičku izvodljivost na osnovu sastava.

        Args:
            formula: Hemijska formula

        Returns:
            Dictionary sa feasibility metrikama
        """
        elements = self.parse_formula_elements(formula)

        # Check for toxic elements (automatic disqualification)
        toxic_count = sum(1 for elem in elements if elem in self.toxic_elements)
        if toxic_count > 0:
            return {
                'feasibility_score': 10,
                'precursor_availability': 10,
                'synthesis_complexity': 10,
                'air_stability': 50,
                'reason': f'Contains toxic element(s): {[e for e in elements if e in self.toxic_elements]}'
            }

        # Precursor availability
        common_count = sum(1 for elem in elements if elem in self.precursor_availability['common'])
        available_count = sum(1 for elem in elements if elem in self.precursor_availability['available'])
        expensive_count = sum(1 for elem in elements if elem in self.precursor_availability['expensive'])
        exotic_count = sum(1 for elem in elements if elem in self.exotic_elements)

        total_elements = len(elements)

        if common_count == total_elements:
            precursor_score = 100
        elif common_count + available_count >= total_elements - 1:
            precursor_score = 80
        elif expensive_count > 0:
            precursor_score = 60
        elif exotic_count > 0:
            precursor_score = 40
        else:
            precursor_score = 70

        # Synthesis complexity (više elemenata = kompleksnije)
        if total_elements <= 4:
            complexity_score = 100
        elif total_elements == 5:
            complexity_score = 85
        elif total_elements == 6:
            complexity_score = 70
        else:
            complexity_score = 50

        # Air stability (oksidi su stabilni)
        if 'O' in elements:
            stability_score = 90
        else:
            stability_score = 50

        # Overall feasibility
        feasibility_score = (0.4 * precursor_score +
                             0.3 * complexity_score +
                             0.3 * stability_score)

        return {
            'feasibility_score': feasibility_score,
            'precursor_availability': precursor_score,
            'synthesis_complexity': complexity_score,
            'air_stability': stability_score,
            'reason': 'Feasible' if feasibility_score > 60 else 'Challenging synthesis'
        }

    def calculate_novelty_score(self, formula: str) -> float:
        """
        Ocenjuje novinu materijala.
        Placeholder - može se proširiti sa literature search API.

        Args:
            formula: Hemijska formula

        Returns:
            Novelty score (0-100)
        """
        # Simplified novelty assessment
        # U pravoj implementaciji, ovde bi bio API call ka naučnim bazama

        elements = self.parse_formula_elements(formula)

        # Više egzotičnih elemenata = verovatno noviji materijal
        exotic_count = sum(1 for elem in elements if elem in self.exotic_elements)
        rare_earth_count = sum(1 for elem in elements if elem in self.rare_earth)

        if exotic_count >= 2:
            return 90  # Highly novel
        elif exotic_count == 1 or rare_earth_count >= 2:
            return 70  # Moderately novel
        elif rare_earth_count == 1:
            return 50  # Some novelty
        else:
            return 40  # Likely well-studied

    def assign_tier(self, total_score: float) -> Tuple[str, str, str]:
        """
        Dodeljuje tier i preporuku na osnovu ukupnog skora.

        Args:
            total_score: Composite score (0-100)

        Returns:
            Tuple (tier, recommendation, rationale)
        """
        if total_score >= 85:
            return ("Tier 1",
                    "STRONGLY RECOMMEND - Priority Synthesis",
                    "Excellent predicted properties, high confidence, feasible synthesis")
        elif total_score >= 75:
            return ("Tier 1",
                    "RECOMMEND - High Priority",
                    "Very good properties, reliable predictions, practical synthesis")
        elif total_score >= 65:
            return ("Tier 2",
                    "CONSIDER - Promising but Risky",
                    "Good potential but higher uncertainty or synthesis challenges")
        elif total_score >= 55:
            return ("Tier 3",
                    "EDGE CASE - Model Validation",
                    "Useful for understanding model limitations, not optimal performance")
        else:
            return ("Tier 4",
                    "NOT RECOMMENDED",
                    "Poor predicted properties, low confidence, or infeasible synthesis")

    def evaluate_candidate(self,
                           formula: str,
                           catboost_pred: float,
                           nn_pred: float,
                           uncertainty: float) -> Dict:
        """
        Glavna funkcija za evaluaciju jednog kandidata.

        Args:
            formula: Hemijska formula
            catboost_pred: CatBoost Dq/B predviđanje
            nn_pred: Neural Network Dq/B predviđanje
            uncertainty: Uncertainty estimate

        Returns:
            Dictionary sa kompletnom evaluacijom
        """
        # Use NN prediction as primary (usually more accurate)
        primary_prediction = nn_pred

        # Calculate all scores
        perf_metrics = self.calculate_performance_score(primary_prediction)
        conf_metrics = self.calculate_confidence_score(catboost_pred, nn_pred, uncertainty)
        feas_metrics = self.calculate_feasibility_score(formula)
        novelty_score = self.calculate_novelty_score(formula)

        # Composite score calculation
        # Weighted combination: performance (40%), confidence (30%), feasibility (20%), novelty (10%)
        total_score = (0.40 * perf_metrics['performance_score'] +
                       0.30 * conf_metrics['confidence_score'] +
                       0.20 * feas_metrics['feasibility_score'] +
                       0.10 * novelty_score)

        # Assign tier and recommendation
        tier, recommendation, rationale = self.assign_tier(total_score)

        # Compile complete evaluation
        evaluation = {
            'Formula': formula,
            'Predicted_DqB': round(primary_prediction, 3),
            'CatBoost_DqB': round(catboost_pred, 3),
            'NN_DqB': round(nn_pred, 3),
            'Uncertainty': round(uncertainty, 3),
            'Predicted_Emission_nm': round(perf_metrics['emission_wavelength'], 1),

            # Scores
            'Total_Score': round(total_score, 1),
            'Performance_Score': round(perf_metrics['performance_score'], 1),
            'Confidence_Score': round(conf_metrics['confidence_score'], 1),
            'Feasibility_Score': round(feas_metrics['feasibility_score'], 1),
            'Novelty_Score': round(novelty_score, 1),

            # Sub-scores
            'Emission_Match': round(perf_metrics['emission_score'], 1),
            'Thermal_Stability': round(perf_metrics['thermal_stability_score'], 1),
            'Model_Agreement': round(conf_metrics['model_agreement_score'], 1),
            'Precursor_Availability': round(feas_metrics['precursor_availability'], 1),

            # Decision
            'Tier': tier,
            'Recommendation': recommendation,
            'Rationale': rationale
        }

        return evaluation

    def evaluate_dataset(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluira kompletan dataset kandidata.

        Args:
            predictions_df: DataFrame sa kolonama [Formula, CatBoost_Pred, NN_Pred, Uncertainty]

        Returns:
            DataFrame sa kompletnom evaluacijom, sortiran po Total_Score
        """
        results = []

        for idx, row in predictions_df.iterrows():
            evaluation = self.evaluate_candidate(
                formula=row['Formula'],
                catboost_pred=row['CatBoost_Pred'],
                nn_pred=row['NN_Pred'],
                uncertainty=row['Uncertainty']
            )
            results.append(evaluation)

        results_df = pd.DataFrame(results)

        # Sort by Total_Score descending
        results_df = results_df.sort_values('Total_Score', ascending=False).reset_index(drop=True)

        return results_df

    def generate_summary_report(self, results_df: pd.DataFrame) -> str:
        """
        Generiše tekstualni izveštaj sa preporukama.

        Args:
            results_df: DataFrame sa evaluacijom

        Returns:
            String sa izveštajem
        """
        report = []
        report.append("=" * 80)
        report.append("EXPERT SYSTEM EVALUATION REPORT")
        report.append("Cr³⁺-Doped Phosphor Candidate Recommendations")
        report.append("=" * 80)
        report.append("")

        # Summary statistics
        tier1 = len(results_df[results_df['Tier'] == 'Tier 1'])
        tier2 = len(results_df[results_df['Tier'] == 'Tier 2'])
        tier3 = len(results_df[results_df['Tier'] == 'Tier 3'])
        tier4 = len(results_df[results_df['Tier'] == 'Tier 4'])

        report.append(f"Total Candidates Evaluated: {len(results_df)}")
        report.append(f"  • Tier 1 (Strongly Recommended): {tier1}")
        report.append(f"  • Tier 2 (Consider): {tier2}")
        report.append(f"  • Tier 3 (Edge Cases): {tier3}")
        report.append(f"  • Tier 4 (Not Recommended): {tier4}")
        report.append("")

        # Top 10 recommendations
        report.append("-" * 80)
        report.append("TOP 10 SYNTHESIS RECOMMENDATIONS")
        report.append("-" * 80)
        report.append("")

        top10 = results_df.head(10)
        for idx, row in top10.iterrows():
            report.append(f"#{idx + 1}. {row['Formula']} (Score: {row['Total_Score']:.1f})")
            report.append(f"    Predicted Dq/B: {row['Predicted_DqB']:.2f} ± {row['Uncertainty']:.2f}")
            report.append(f"    Predicted Emission: {row['Predicted_Emission_nm']:.0f} nm")
            report.append(f"    {row['Tier']}: {row['Recommendation']}")
            report.append(f"    Rationale: {row['Rationale']}")
            report.append("")

        return "\n".join(report)


def main_expert_system_pipeline(predictions_df: pd.DataFrame,
                                output_excel: str = 'expert_system_recommendations.xlsx',
                                output_report: str = 'expert_system_report.txt') -> pd.DataFrame:
    """
    Glavna funkcija pipeline-a ekspertskog sistema.

    Args:
        predictions_df: DataFrame sa ML predikcijama
        output_excel: Putanja za Excel output
        output_report: Putanja za tekstualni report

    Returns:
        DataFrame sa evaluacijom
    """
    # Initialize expert system
    expert = PhosphorExpertSystem(
        target_dqb_range=(2.8, 3.8),
        target_emission_range=(650, 720)
    )

    # Evaluate all candidates
    print("Evaluating candidates with expert system...")
    results_df = expert.evaluate_dataset(predictions_df)

    # Save Excel report
    results_df.to_excel(output_excel, index=False)
    print(f"✓ Saved detailed evaluation to: {output_excel}")

    # Generate and save text report
    report = expert.generate_summary_report(results_df)
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✓ Saved summary report to: {output_report}")

    # Print summary to console
    print("\n" + report)

    return results_df


# Example usage
if __name__ == "__main__":
    # Example data structure (replace with actual predictions from your models)
    example_data = pd.DataFrame({
        'Formula': ['Ca2MgWO6', 'Sr2ScNbO6', 'La2MgTiO6', 'Ba2MgWO6', 'CdWO4'],
        'CatBoost_Pred': [3.15, 3.28, 2.95, 3.42, 2.65],
        'NN_Pred': [3.18, 3.25, 2.98, 3.45, 2.70],
        'Uncertainty': [0.08, 0.12, 0.18, 0.10, 0.25]
    })

    # Run expert system
    results = main_expert_system_pipeline(
        predictions_df=example_data,
        output_excel='expert_system_recommendations.xlsx',
        output_report='expert_system_report.txt'
    )

    print("\n✓ Expert system evaluation complete!")

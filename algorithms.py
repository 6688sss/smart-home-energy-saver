from typing import Dict, List, Tuple


def hill_climbing_schedule(
    schedules: Dict[int, Dict[str, object]],
    start_id: int = 1,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """
    Локальный поиск Hill Climbing для минимизации стоимости энергопотребления
    среди 5 упорядоченных слотов расписания.
    """
    current_id = start_id
    steps: List[Dict[str, object]] = []

    while True:
        current_item = schedules[current_id]
        neighbors = [i for i in (current_id - 1, current_id + 1) if i in schedules]
        best_neighbor_id = min(neighbors, key=lambda idx: float(schedules[idx]["cost"]))
        best_neighbor = schedules[best_neighbor_id]

        steps.append(
            {
                "current_slot": current_item["name"],
                "current_cost": float(current_item["cost"]),
                "best_neighbor": best_neighbor["name"],
                "neighbor_cost": float(best_neighbor["cost"]),
                "moved": float(best_neighbor["cost"]) < float(current_item["cost"]),
            }
        )

        if float(best_neighbor["cost"]) < float(current_item["cost"]):
            current_id = best_neighbor_id
        else:
            break

    return schedules[current_id], steps


def bayesian_occupancy_update(
    prior_occupied: float,
    p_alert_given_occupied: float,
    p_alert_given_unoccupied: float,
    sensor_alert: bool = True,
) -> Dict[str, float]:
    """
    Байесовское обновление вероятности присутствия человека в комнате
    по показанию датчика (Alert = True / No Alert = False).
    """
    p_occ = prior_occupied
    p_unocc = 1.0 - prior_occupied

    if sensor_alert:
        p_obs_given_occ = p_alert_given_occupied
        p_obs_given_unocc = p_alert_given_unoccupied
    else:
        p_obs_given_occ = 1.0 - p_alert_given_occupied
        p_obs_given_unocc = 1.0 - p_alert_given_unoccupied

    p_evidence = (p_obs_given_occ * p_occ) + (p_obs_given_unocc * p_unocc)
    posterior_occ = (p_obs_given_occ * p_occ) / p_evidence if p_evidence > 0 else p_occ

    return {
        "prior_occupied": p_occ,
        "p_evidence": p_evidence,
        "posterior_occupied": posterior_occ,
        "posterior_unoccupied": 1.0 - posterior_occ,
    }


def mdp_appliance_decision(
    posterior_occupied: float,
    rewards: Dict[str, Dict[str, float]],
) -> Tuple[str, Dict[str, float]]:
    """
    Одношаговая оценка ожидаемой полезности действий в MDP:
    EU(a) = P(Occupied)*R(a, Occupied) + P(Unoccupied)*R(a, Unoccupied).
    """
    p_occ = posterior_occupied
    p_unocc = 1.0 - posterior_occupied

    expected_utilities: Dict[str, float] = {}
    for action, state_rewards in rewards.items():
        eu = (p_occ * state_rewards["Occupied"]) + (p_unocc * state_rewards["Unoccupied"])
        expected_utilities[action] = eu

    best_action = max(expected_utilities, key=lambda act: expected_utilities[act])
    return best_action, expected_utilities

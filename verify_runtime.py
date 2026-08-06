import py_compile
from core.enums import MovementType, PhaseType
from core.intersection import Intersection
from config import phases as phase_config, traffic_profiles as tp

modules = [
    'core.enums', 'core.approach', 'core.movement', 'core.intersection',
    'config.phases', 'config.traffic_profiles', 'analytics.statistics', 'simulation'
]
for module in modules:
    py_compile.compile(module.replace('.', '\\') + '.py', doraise=True)

intersection = Intersection()
all_ids = {m.movement_id for m in intersection.all_movements()}
expected_ids = {f"{a}_{mt.name}" for a in ('North', 'South', 'East', 'West') for mt in MovementType}
print('all_movements_count=', len(all_ids), 'expected=', len(expected_ids))
assert all_ids == expected_ids, 'Movement set mismatch'

plan = phase_config.build_phase_plan(intersection)
expected_phases = {
    PhaseType.PHASE_1,
    PhaseType.PHASE_2,
    PhaseType.PHASE_3,
    PhaseType.PHASE_4,
    PhaseType.PHASE_5,
    PhaseType.PHASE_6,
    PhaseType.PHASE_7,
    PhaseType.PHASE_8,
    PhaseType.PHASE_9,
    PhaseType.PHASE_10,
}
assert set(plan.keys()) == expected_phases, 'Phase plan keys mismatch'

OFFICIAL = {
    PhaseType.PHASE_1: {'West_STRAIGHT','West_LEFT','West_UTURN','North_LEFT','East_LEFT','South_RIGHT','South_LEFT'},
    PhaseType.PHASE_2: {'North_STRAIGHT','North_UTURN','North_LEFT','East_LEFT','South_LEFT','West_LEFT','West_RIGHT'},
    PhaseType.PHASE_3: {'East_STRAIGHT','East_UTURN','East_LEFT','South_LEFT','West_LEFT','North_LEFT','North_RIGHT'},
    PhaseType.PHASE_4: {'South_STRAIGHT','South_UTURN','South_LEFT','West_LEFT','North_LEFT','East_RIGHT','East_LEFT'},
    PhaseType.PHASE_5: {'West_STRAIGHT','West_LEFT','West_UTURN','North_LEFT','East_UTURN','East_STRAIGHT','East_RIGHT','South_LEFT'},
    PhaseType.PHASE_6: {'South_STRAIGHT','South_LEFT','South_UTURN','West_LEFT','North_STRAIGHT','North_UTURN','North_LEFT','East_LEFT'},
    PhaseType.PHASE_7: {'South_STRAIGHT','South_LEFT','South_RIGHT','South_UTURN','West_LEFT','North_LEFT','East_UTURN','East_LEFT'},
    PhaseType.PHASE_8: {'West_STRAIGHT','West_LEFT','West_RIGHT','West_UTURN','North_LEFT','East_LEFT','South_LEFT','South_UTURN'},
    PhaseType.PHASE_9: {'North_STRAIGHT','North_LEFT','North_RIGHT','North_UTURN','East_LEFT','South_LEFT','West_LEFT','West_UTURN'},
    PhaseType.PHASE_10: {'East_STRAIGHT','East_UTURN','East_RIGHT','East_LEFT','South_LEFT','West_LEFT','North_UTURN','North_LEFT'},
}
for pt, official in OFFICIAL.items():
    actual = {m.movement_id for m in plan[pt].movements}
    assert actual == official, f'{pt.name} mismatch: missing={sorted(official-actual)} extra={sorted(actual-official)}'

em = phase_config.build_emergency_phase(intersection, 'North')
assert em.phase_type == PhaseType.EMERGENCY_OVERRIDE, 'Emergency phase type wrong'
assert set(em.movements) == set(intersection.get_approach('North').movements.values()), 'Emergency movements mismatch'
assert PhaseType.EMERGENCY_OVERRIDE not in plan, 'Emergency override leaked into normal plan'

keys = tp.all_movement_keys()
assert len(keys) == 16, 'Traffic profile movement key count wrong'
rates = tp._base_rates(0.25, 0.25, 0.20, 0.20, 0.12, 0.12, 0.10, 0.10, 0.08, 0.08, 0.06, 0.06)
assert rates['North.UTURN'] == 0.25 * 0.05, 'UTurn default rate mismatch'
print('ALL CHECKS PASSED')

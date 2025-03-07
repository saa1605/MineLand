import mineland

mland = mineland.make(
    task_id="survival_0.01_days",
    agents_count = 2,
)

obs = mland.reset()

for i in range(5000):
    act = mineland.Action.no_op(len(obs))
    obs, code_info, event, done, task_info = mland.step(action=act)
    if done: break

mland.close()
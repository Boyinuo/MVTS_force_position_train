"""Build a standalone, dependency-free HTML audit of the exposure experiments."""
import html
import json
from pathlib import Path


def table(headers, rows):
    return '<table><tr>' + ''.join(f'<th>{html.escape(str(h))}</th>' for h in headers) + '</tr>' + ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(str(c))}</td>' for c in row) + '</tr>' for row in rows) + '</table>'


def linechart(series):
    colors = ['#7f8da4', '#126b55', '#c76a35', '#7059a6']
    top = max(max(v) for v in series.values()) * 1.12
    points = lambda v: ' '.join(f'{60+i*65},{300-x/top*240:.1f}' for i,x in enumerate(v))
    svg = '<svg viewBox="0 0 720 365" role="img" aria-label="同一图像复用期间子事件XY误差">'
    for y in range(5):
        value = top*y/4
        sy = 300-value/top*240
        svg += f'<line x1="60" x2="660" y1="{sy}" y2="{sy}" stroke="#e1e6ed"/><text x="12" y="{sy+5}" font-size="13">{value:.1f}</text>'
    for i in range(10):
        svg += f'<text x="{56+i*65}" y="323" font-size="13">{i}</text>'
    for (name, values), color in zip(series.items(), colors):
        svg += f'<polyline points="{points(values)}" fill="none" stroke="{color}" stroke-width="3"/>'
    for i, ((name,_),color) in enumerate(zip(series.items(),colors)):
        svg += f'<text x="{60+i*190}" y="355" fill="{color}" font-size="13">{html.escape(name)}</text>'
    return svg + '</svg>'


def main():
    root = Path('runs/image_exposure_v1')
    s = json.loads((root/'summary.json').read_text())
    prep = json.loads((root/'prepared.json').read_text())
    diag = json.loads((root/'image_exposure_timing_diagnostics.json').read_text())
    baseline = s['baseline']['xy_mean_mm']
    names = {'image_exposure_time_control': '图像年龄模型：无曝光偏移',
             'image_exposure_anchor_control': '读图时刻锚点监督：无曝光偏移',
             'image_exposure_midpoint': '曝光中点监督：7.8 ms',
             'image_exposure_average': '15.6 ms曝光平均监督',
             'image_exposure_full_delay': '总偏移15.6 ms敏感性对照'}
    rows = [['原始两数据集纯视觉基线', f'{s["selection"]["baseline_val"]["xy_mean_mm"]:.4f}', f'{baseline:.4f}', '—']]
    for key, value in s['models'].items():
        score = value['xy_mean_mm']
        rows.append([names.get(key,key), f'{s["selection"]["variants"][key]["validation_xy_mm"]:.4f}', f'{score:.4f}', f'{(baseline-score)/baseline*100:+.2f}%'])
    rows.append(['仅模型预测运动补偿（验证集选7.8 ms）', f'{s["selection"]["selected_compensation"]["val_xy_mean_mm"]:.4f}',
                 f'{s["selected_physics_compensation"]["xy_mean_mm"]:.4f}',
                 f'{(baseline-s["selected_physics_compensation"]["xy_mean_mm"])/baseline*100:+.2f}%'])
    content = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>曝光时间与位置精度验证</title><style>
    body{max-width:1060px;margin:42px auto;padding:0 24px;color:#203044;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;background:#f7f9fc}
    h1{font-size:30px} h2{font-size:23px;margin-top:36px} .card{background:white;border:1px solid #dce4ed;border-radius:12px;padding:22px;margin:20px 0}
    table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{border:1px solid #dce4ed;padding:10px;text-align:left}th{background:#edf2f7}code{background:#edf2f7;padding:2px 6px}svg{width:100%;max-width:760px} .note{color:#805728} a{color:#126b55}</style>
    <h1>15.6 ms曝光时间：位置预测实验</h1><p>2026-11-24 · 仅使用v6滑移与折线 · 原始模型/缓存/数据保持只读</p>'''
    content += '<div class="card"><b>实验原则：</b>按原checkpoint保存的轨迹划分，训练/验证/测试图像窗口为1891/443/368，轨迹为252/59/49。测试含3680个相关子事件，不能当作3680个独立图像。主指标是原始当前时刻XY欧氏距离均值，单位mm；所有方法使用同一真值。<br>曝光中点前移7.8 ms仅是零额外读出延迟下的假设。已知曝光长度不能确定总采集延迟。没有将机械臂位置或真实速度作为预测输入。</div>'
    content += '<h2>第一轮：冻结原视觉特征，训练位置与运动模型</h2>' + table(['方法','验证XY误差','测试XY误差','相对基线改善'],rows)
    content += '<p>五个新位置模型均训练30轮，seed=42，batch=8条独立轨迹，lr=0.001，原视觉主干与注意力冻结。损失为当前位置监督，另按对照添加锚点位置监督。此训练与原三头网络30轮全量训练不同。验证集选择了无曝光偏移的锚点对照；其测试结果未超过原基线。</p>'
    content += '<h2>误差随图像复用时间增长</h2>' + linechart({'原始基线':s['baseline']['by_subevent_xy_mean_mm'],
        '7.8ms曝光中点模型':s['models']['image_exposure_midpoint']['by_subevent_xy_mean_mm']})
    content += '<p>子事件0使用刚读完的图像，子事件9使用同一图像约60 ms以后。曝光延迟与图像复用造成的陈旧程度同时存在。</p>'
    content += '<h2>验证集诊断</h2><p>将原预测与更早的真值比较，误差在候选23.4 ms处最低，但这只是残余时序滞后的诊断，不能算作当前时刻精度提升，也不能当作相机硬件延迟测量。7.8 ms对应的验证集实际XY位移均值约0.255 mm，P95约0.806 mm。</p>'
    v2 = Path('runs/image_exposure_anchor_v2/summary.json')
    if v2.exists():
        content += '<h2>第二轮：完整视觉分支微调</h2><p>详细训练配置、验证集选择、所有测试结果见<a href="../image_exposure_anchor_v2/summary.json">第二轮summary.json</a>。单帧模型的输出锚点对应估计图像物理时刻；向当前时刻的外推只使用相邻预测的位置差。</p>'
        value=json.loads(v2.read_text())
        second_names = {'image_exposure_nooffset_anchor':'读图时刻监督：零偏移对照',
            'image_exposure_midpoint_anchor':'估计曝光中点：7.8 ms',
            'image_exposure_average_anchor':'15.6 ms曝光平均：中点7.8 ms',
            'image_exposure_estimated_delay_anchor':'估计总延迟：23.4 ms'}
        rows=[]
        for key, score in value['models'].items():
            config=value['selection']['variants'][key]
            rows.append([second_names.get(key,key),config['best_epoch'],config['settings']['smoothing'],
                f'{config["validation_xy_mm"]:.4f}',f'{score["xy_mean_mm"]:.4f}',
                f'{(baseline-score["xy_mean_mm"])/baseline*100:+.2f}%'])
        content += table(['微调方案','选中epoch','运动EMA权重','验证XY/mm','测试XY/mm','较原基线改善'],rows)
        selected=value['primary_selected_model']
        score=value['baseline'] if selected=='original_baseline' else value['models'][selected]
        ci=value['paired_trajectory_bootstrap']['baseline_minus_candidate_xy_mm_95ci']
        content += f'<div class="card">验证集选中：<b>{html.escape(second_names.get(selected,selected))}</b>。其测试XY误差为<b>{score["xy_mean_mm"]:.4f} mm</b>，较原基线改善{(baseline-score["xy_mean_mm"])/baseline*100:+.2f}%。整条轨迹bootstrap的改善95%区间为[{ci[0]:.4f}, {ci[1]:.4f}] mm，正值表示改善。此区间不包含训练随机种子的变化。</div>'
        content += '<p><b>归因结论：</b>本轮验证集选中零曝光偏移对照，因此不能将收益归因于已知15.6 ms曝光。曝光中点与曝光平均均未超过匹配的零偏移方案。新模型的直线滑移改善更大，但折线、P95和Z误差有退化，不建议直接全量替换原部署模型。</p>'
        arrival=Path('runs/image_exposure_anchor_v2/image_exposure_image_arrival_metrics.json')
        if arrival.exists():
            a=json.loads(arrival.read_text())
            content += f'<p><b>新图像到达时的定位：</b>每张图的第0子事件，原基线XY误差{a["baseline"]["xy_mean_mm"]:.4f} mm，新模型{a["selected"]["xy_mean_mm"]:.4f} mm，改善{a["improvement_percent"]:.1f}%。这是读图完成时刻的定位，仍不是曝光物理时刻的定位。该分项是描述性次指标；后续复用同帧的误差会增长。</p>'
        content += '<p>本轮从原已训练模型初始化，按位置监督微调全部视觉、注意力、GRU和位置头；不是从随机权重训练。每张图重置位置GRU，外层维护相邻图像预测的位置差与时间。每个方案30轮，batch=8图像，lr=0.00001，seed=42。未继续训练力与滑移头。</p>'
        if selected!='original_baseline':
            content += linechart({'原始基线':value['baseline']['by_subevent_xy_mean_mm'],
                '验证集选中模型':score['by_subevent_xy_mean_mm']})
        content += '<p>推理使用独立接口<code>image_exposure_experiments.predict.ImageExposurePredictor</code>，不要直接覆盖原部署权重；接口说明见代码目录内image_exposure_README.txt。</p>'
    content += '<h2>边界与复现</h2><p class="note">第二轮是在查看第一轮测试结果之后设计，因此复用测试集上的后续结果属于探索性证据。最终确认需要新的独立采集轨迹。原事件网格由已记录的相邻图像区间十等分，本实验验证的是该网格上的回放误差，并非实测在线150Hz调度延迟。</p>'
    content += '<p>曝光平均标签采用原始实测位置轨迹的分段线性精确积分，只作监督近似。相机成像、材料形变、滚动快门和传输缓存可能带来额外差异。所有查询均在实测范围内，未做外推。</p>'
    content += '<p>复现命令（仓库根目录）：<br><code>.venv/Scripts/python.exe -m image_exposure_experiments.prepare --output runs/image_exposure_NEW</code><br><code>.venv/Scripts/python.exe -m image_exposure_experiments.run --output runs/image_exposure_NEW</code><br><code>.venv/Scripts/python.exe -m pytest image_exposure_experiments -q -p no:cacheprovider</code></p>'
    content += '<p>原始checkpoint、两个缓存manifest、原模型源码的SHA256在实验前后核对均一致。所有新目录均使用image_exposure_前缀。</p></html>'
    (root/'image_exposure_report.html').write_text(content, encoding='utf-8')
    print(root/'image_exposure_report.html')


if __name__ == '__main__':
    main()

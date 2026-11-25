"""Generate the independent multimodal contact/slip experiment report."""
import argparse
import html
import json
from pathlib import Path


def table(headers, rows):
    return '<table><tr>'+''.join(f'<th>{html.escape(str(x))}</th>' for x in headers)+'</tr>'+''.join(
        '<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</table>'


def lines(series):
    ymax=max(max(v) for v in series.values())*1.1
    svg='<svg viewBox="0 0 760 365" role="img" aria-label="每个子事件的位置误差">'
    for j in range(5):
        y=300-j*60
        svg+=f'<line x1="50" x2="700" y1="{y}" y2="{y}" stroke="#e3e9ef"/><text x="2" y="{y+4}" font-size="13">{j*ymax/4:.1f}</text>'
    for i in range(10):svg+=f'<text x="{48+i*70}" y="322" font-size="13">{i}</text>'
    for j,((name,values),color) in enumerate(zip(series.items(),['#748095','#197461','#c37435'])):
        points=' '.join(f'{50+i*70},{300-v/ymax*240:.2f}' for i,v in enumerate(values))
        svg+=f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/><text x="{50+j*225}" y="354" fill="{color}" font-size="13">{name}</text>'
    return svg+'</svg>'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='runs/image_exposure_gated_compare')
    args=parser.parse_args();root=Path(args.output)
    s=json.loads((root/'summary.json').read_text())
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Contact / Slip 单图位置方法验证</title><style>
    body{max-width:1100px;margin:40px auto;padding:0 24px;background:#f7f9fc;color:#223447;font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif}
    h1{font-size:29px}h2{font-size:23px;margin-top:36px}table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{border:1px solid #dbe4ed;padding:10px;text-align:left}th{background:#edf2f6}.card{padding:20px;border:1px solid #dbe4ed;border-radius:12px;background:white;margin:20px 0}svg{max-width:780px;width:100%}code{background:#e8eef4;padding:3px 5px}a{color:#197461}</style>
    <h1>model_contact_gated / model_slip_gated：单图微调方法验证</h1><p>2026-11-24 · 仅滑移与折线 · 新模型独立保存，原部署权重和数据未修改</p>'''
    text+='<div class="card"><b>比较对象：</b>现有部署模型曾训练过按压数据，因此本次从仅含两数据集的多模态三头模型迁移共有权重，建立新的contact/slip架构基线。结果回答的是这两种架构的独立受控实验，不是现有含按压数据部署checkpoint的直接升级效果。<br><b>共同条件：</b>真实图像和真实因果音频，禁用第5路麦克风。声学编码器与投影固定，缓存其实际编码以加速；视觉、融合、GRU和预测头按各实验训练。架构在正常推理时仍保留声学编码。<br><b>数据划分：</b>1891/443/368个训练/验证/测试窗口，252/59/49条轨迹。测试3680个相关子事件。标签和划分保持原样，没有把更早真值当作当前时刻精度改善。</div>'
    concise={}
    for model,r in s['results'].items():
        sel=s['selection'][model]
        baseline=r['baseline']['raw_position'];arrival=r['baseline']['image_arrival']
        text+=f'<h2>{html.escape(model)}</h2>'
        rows=[['连续GRU基线',f'{sel["baseline"]["validation_xy_mm"]:.4f}',f'{baseline["xy_mean_mm"]:.4f}',f'{arrival["xy_mean_mm"]:.4f}',f'{baseline["xy_p95_mm"]:.4f}',f'{baseline["mae_mm"]["z"]:.4f}','—']]
        for key,name in [('image_exposure_nooffset','单图微调：零偏移'),('image_exposure_midpoint','单图微调：7.8ms中点')]:
            v=r['variants'][key];m=v['raw_position'];c=sel['variants'][key]
            gain=(1-m['xy_mean_mm']/baseline['xy_mean_mm'])*100
            rows.append([name,f'{c["validation_xy_mm"]:.4f}',f'{m["xy_mean_mm"]:.4f}',f'{v["image_arrival"]["xy_mean_mm"]:.4f}',f'{m["xy_p95_mm"]:.4f}',f'{m["mae_mm"]["z"]:.4f}',f'{gain:+.2f}%'])
        text+=table(['方案','验证XY/mm','测试全部事件XY/mm','图像到达XY/mm','测试P95/mm','Z MAE/mm','总体改善'],rows)
        selected=sel['selected']
        selected_score=baseline['xy_mean_mm'] if selected=='baseline' else r['variants'][selected]['raw_position']['xy_mean_mm']
        concise[model]={'selected_by_validation':selected,'baseline_test_xy_mm':baseline['xy_mean_mm'],
            'selected_test_xy_mm':selected_score,'overall_gain_percent':(1-selected_score/baseline['xy_mean_mm'])*100}
        text+=f'<p>验证集选择：<b>{html.escape(selected)}</b>。连续基线先训练30轮；单图方案各从同一个选中基线再微调30轮，lr=1e-5。所有checkpoint/运动平滑参数在测试之前固定。单图方法每张图重置GRU，仅使用该图到达时的第一个音频窗口，随后使用相邻预测的位置差外推；后续9个音频窗口不参与单图方案，不能宣称保留150Hz声学更新能力。</p>'
        series={'连续基线':baseline['by_subevent_xy_mean_mm']}
        for key,name in [('image_exposure_nooffset','单图零偏移'),('image_exposure_midpoint','单图7.8ms')]:series[name]=r['variants'][key]['raw_position']['by_subevent_xy_mean_mm']
        text+=lines(series)
        rows=[]
        for key,name in [('image_exposure_nooffset','零偏移'),('image_exposure_midpoint','7.8ms')]:
            v=r['variants'][key]
            ci=v['paired_bootstrap']['baseline_minus_candidate_xy_mm_95ci']
            a=v['image_arrival_bootstrap']['baseline_minus_candidate_xy_mm_95ci']
            rows.append([name,f'[{ci[0]:.4f}, {ci[1]:.4f}]',f'[{a[0]:.4f}, {a[1]:.4f}]'])
        text+=table(['方案','总体改善95%区间/mm','图像到达改善95%区间/mm'],rows)
        text+='<p>区间按整条轨迹配对bootstrap计算；正值表示改善。只覆盖轨迹抽样不确定性，不覆盖训练随机种子变化。</p>'
        if model=='model_contact_gated':
            rows=[]
            for name,v in [('基线',r['baseline']),*list(r['variants'].items())]:
                rows.append([name,f'{v["gated_position"]["xy_mean_mm"]:.4f}',f'{v["position_output_coverage"]*100:.2f}%',v['contact_false_negative_events']])
            text+=table(['方案','门控后XY/mm','位置输出覆盖率','接触漏检子事件'],rows)
            text+='<p>门控拒绝的样本仍计入误差，没有按预测有效性剔除。所有数据都是接触正样本，特异性、误报率与无接触识别能力无法评估。单图方案接触迟滞状态每图更新并保持，原连续基线每子事件更新。</p>'
        rows=[]
        for name,v in [('基线',r['baseline']),*list(r['variants'].items())]:
            rows.append([name,f'{v["first_event_force_mae"][2]:.5f}',f'{v["first_event_slip_mae"][0]:.5f}',f'{v["first_event_slip_mae"][1]:.5f}'])
        text+=table(['方案：仅图像到达首事件','Fz MAE/N','Δsx MAE/mm','Δsy MAE/mm'],rows)
        text+='<p>位置微调会改变共享特征，即使其他输出头固定，也不保证力、滑移和接触输出不变。此处其他输出仅比较首事件，不能与原模型全部150Hz事件指标混为一谈。</p>'
    text+='<h2>复现与边界</h2><p><code>.venv/Scripts/python.exe -m image_exposure_experiments.gated_compare --output runs/image_exposure_gated_NEW</code><br><code>.venv/Scripts/python.exe -m image_exposure_experiments.gated_report --output runs/image_exposure_gated_NEW</code></p>'
    text+='<p>源checkpoint仅在两个指定缓存上训练；没有读取按压训练样本或加载按压预训练权重。正常的单图推理需要每图重置GRU、使用首个真实因果音频与实际事件delta_t，按anchor_config维护预测运动历史；不能把独立权重直接覆盖原部署best.pt。baseline checkpoint中的args保留共有权重来源的历史信息，本次实际配置以protocol.json与anchor_config为准。</p>'
    text+='<p>这是同一已使用测试集上的单seed探索实验。即使有改善，也需要新轨迹和多个训练seed确认。15.6ms曝光持续时间并不等于总采集延迟；7.8ms仅是无额外读出延迟的中点假设。所有原始关键文件SHA256均在前后核对一致。</p></html>'
    (root/'image_exposure_report.html').write_text(text,encoding='utf-8')
    (root/'image_exposure_conclusions.json').write_text(json.dumps(concise,indent=2))
    print(json.dumps(concise,indent=2))


if __name__=='__main__':main()

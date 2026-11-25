from types import SimpleNamespace

import numpy as np
import torch

from image_exposure_experiments.multimodal_anchor import _first_forward, first_event_metrics


class Recorder:
    requires_timing = True

    def forward_sequence(self,image,audio,hidden=None,**kwargs):
        self.received=(image,audio,hidden,kwargs)
        return {'position':audio.mean((2,3,4))[...,None].expand(-1,-1,3)}


def test_anchor_cannot_consume_later_audio_and_uses_event_dt():
    model=Recorder()
    inputs={'image':torch.zeros(2,3,224,224),'audio':torch.randn(2,10,6,64,28),
            'delta_t':torch.full((2,10),.006),'image_age':torch.zeros(2,10)}
    a=_first_forward(model,inputs,7.8)['position']
    inputs['audio'][:,1:]=9999
    b=_first_forward(model,inputs,7.8)['position']
    torch.testing.assert_close(a,b)
    _,audio,hidden,kwargs=model.received
    assert audio.shape[1]==1 and hidden is None
    torch.testing.assert_close(kwargs['delta_t'],torch.full((2,1),.006))
    torch.testing.assert_close(kwargs['image_age'],torch.full((2,1),.0078))
    assert not kwargs['has_history'].any()


def test_contact_rejections_still_count_as_position_error():
    predicted={'raw_first':np.ones((2,11)),'contact_probability':np.array([.1,.1])}
    data={'target':np.ones((2,10,11)),'target_valid':np.ones((2,10,11),bool),
          'contact':np.ones((2,10),bool),'contact_valid':np.ones((2,10),bool),
          'records':[{'sequence':'a','order':0},{'sequence':'a','order':1}]}
    result=first_event_metrics(predicted,data)
    assert result['contact']['fn']==2
    assert result['position_output_coverage']==0
    assert np.isclose(result['gated_first_xy_mean_mm'],np.sqrt(2))


def test_hysteresis_retains_contact_between_image_updates():
    predicted={'raw_first':np.ones((3,11)),'contact_probability':np.array([.7,.45,.35])}
    data={'target':np.ones((3,10,11)),
          'records':[{'sequence':'a','order':j} for j in range(3)]}
    result=first_event_metrics(predicted,data)
    assert result['contact']['tp']==2 and result['contact']['fn']==1

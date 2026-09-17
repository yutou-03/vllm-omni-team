from vllm_omni.scheduling.admission_probe import choose


def candidate(name, deadline, occupancy, order):
    return {'id':name,'order':order,'deadlines':{'ttft':deadline,'e2e':100},
            'costs':{'ttft':.1,'e2e':occupancy},'occupancy':occupancy}


def test_short_first_can_save_both_with_sixty_three_resident_requests():
    cs=[candidate('long',1,5,0),candidate('short',2,.3,1)]
    releases=[0]+[10]*63
    assert choose(cs,releases,now=0,mode='milestone')[0]=='long'
    assert choose(cs,releases,now=0,mode='occupancy')[0]=='short'
    assert choose(cs,releases,now=0,mode='no_occupancy')[0]=='long'


def test_tight_long_deadline_prevents_short_first():
    cs=[candidate('long',.2,5,0),candidate('short',10,.3,1)]
    assert choose(cs,[0]+[10]*63,now=0,mode='occupancy')[0]=='long'
    assert choose(cs,[0]+[10]*63,now=0,mode='shortest')[0]=='short'


def test_other_resident_slot_release_eliminates_conflict():
    cs=[candidate('long',1,5,0),candidate('short',2,.3,1)]
    assert choose(cs,[0,.2]+[10]*62,now=0,mode='occupancy')[0]=='long'


def test_two_free_slots_eliminate_reordering():
    cs=[candidate('long',1,5,0),candidate('short',2,.3,1)]
    assert choose(cs,[0,0]+[10]*62,now=0,mode='occupancy')[0]=='long'

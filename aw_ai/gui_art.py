"""Small original vector sprites, drawn at any board zoom without assets."""

def terrain(c, tile, x, y, size, owner=None, links=()):
    def rect(a,b,d,e,fill,outline=''):
        c.create_rectangle(x+a*size,y+b*size,x+d*size,y+e*size,fill=fill,outline=outline)
    def poly(points,fill):
        c.create_polygon(*[v for a,b in points for v in (x+a*size,y+b*size)],fill=fill,outline='')
    def line(points,fill,width=1):
        c.create_line(*[v for a,b in points for v in (x+a*size,y+b*size)],fill=fill,width=max(1,size*width))
    color=('#467dbe','#d55b50')[owner] if owner is not None else '#929a96'
    if tile in ('plain','forest'):
        for a,b in ((.15,.2),(.7,.7),(.3,.85)):
            line(((a,b),(a+.04,b-.06),(a+.08,b)), '#a2bc88',.018)
    if tile=='forest':
        for a,b in ((.27,.52),(.68,.57),(.47,.84)):
            rect(a-.025,b-.04,a+.025,b+.08,'#79583e')
            poly(((a-.2,b),(a,b-.45),(a+.2,b)),'#315d43')
            poly(((a-.14,b-.08),(a,b-.4),(a+.05,b-.08)),'#568357')
    elif tile=='mountain':
        poly(((.04,.87),(.42,.13),(.94,.87)),'#7b827c')
        poly(((.42,.13),(.54,.87),(.94,.87)),'#596660')
        poly(((.26,.41),(.42,.13),(.61,.42),(.46,.34),(.38,.45)),'#e3e8df')
    elif tile in ('water','river','shoal'):
        for a,b in ((.1,.25),(.53,.51),(.16,.8)):
            line(((a,b),(a+.1,b+.03),(a+.23,b)), '#b6d7df',.025)
    elif tile in ('road','bridge'):
        if not links: links=((0,-1),(0,1))
        rect(.32,.32,.68,.68,'#969b91')
        for dx,dy in links:
            line(((.5,.5),(.5+.5*dx,.5+.5*dy)),'#969b91',.34)
            line(((.5+.18*dx,.5+.18*dy),(.5+.36*dx,.5+.36*dy)),'#e3dfbc',.025)
        if tile=='bridge':
            rect(.24,0,.29,1,'#786e58'); rect(.71,0,.76,1,'#786e58')
    elif tile in ('city','factory','airport','hq','tower'):
        rect(.12,.8,.91,.91,'#929d8a')
        if tile=='airport':
            rect(.14,.33,.88,.79,'#747f7b')
            for a in (.22,.42,.62): rect(a,.55,a+.1,.58,'#e9e9d2')
            rect(.61,.17,.82,.45,'#d4d5bd'); rect(.6,.15,.84,.24,color)
        elif tile=='tower':
            line(((.3,.85),(.5,.25),(.7,.85)),'#626f70',.065)
            line(((.35,.64),(.64,.64)),'#e6e6d3',.04)
            line(((.5,.3),(.5,.12)),color,.05)
            c.create_oval(x+.37*size,y+.12*size,x+.63*size,y+.35*size,outline=color,width=max(1,size*.06))
        else:
            rect(.2,.37,.78,.83,'#ebdfb5','#657369')
            rect(.59,.37,.78,.83,'#c9bf9c')
            if tile=='factory':
                rect(.65,.1,.76,.47,'#747d76')
                poly(((.16,.43),(.16,.25),(.37,.36),(.37,.23),(.57,.36),(.8,.36),(.8,.45)),color)
                rect(.27,.6,.48,.83,'#576767')
            else:
                poly(((.13,.39),(.48,.15),(.85,.39)),color)
                rect(.42,.61,.56,.83,'#66726c')
                if tile=='hq':
                    rect(.37,.16,.58,.37,'#e7dfbe'); rect(.35,.12,.6,.2,color)
            for a in (.27,.62): rect(a,.48,a+.09,.57,'#7dacae')
        line(((.87,.72),(.87,.17)),'#566a60',.02)
        poly(((.87,.18),(.99,.22),(.87,.31)),color)


class MirroredCanvas:
    """Reflect only the sprite drawing; labels and terrain remain upright."""
    def __init__(self, canvas, left, size):
        self.canvas, self.axis = canvas, 2*left+size

    def __getattr__(self, name):
        method=getattr(self.canvas,name)
        def draw(*coordinates,**style):
            xy=list(coordinates)
            xy[::2]=[self.axis-x for x in xy[::2]]
            if name in ('create_rectangle','create_oval'):
                xy[0],xy[2]=min(xy[0],xy[2]),max(xy[0],xy[2])
            return method(*xy,**style)
        return draw


def unit(c, kind, x, y, size, owner, acted=False, flash=False):
    """Blue faces right, Red faces left, independent of human/AI control."""
    if owner == 1: c=MirroredCanvas(c,x,size)
    body=('#396fb0','#c74840')[owner]
    light=('#86b8e4','#f2957b')[owner]
    dark='#263e47'
    if acted: body=('#7f97ad','#ac8a83')[owner]; light='#c1c6bd'
    if flash: body=light='#fffbd9'
    def rect(a,b,d,e,fill): c.create_rectangle(x+a*size,y+b*size,x+d*size,y+e*size,fill=fill,outline='')
    def poly(points,fill): c.create_polygon(*[v for a,b in points for v in (x+a*size,y+b*size)],fill=fill,outline=dark,width=max(1,size*.015))
    def oval(a,b,d,e,fill): c.create_oval(x+a*size,y+b*size,x+d*size,y+e*size,fill=fill,outline='')
    def line(points,fill,width): c.create_line(*[v for a,b in points for v in (x+a*size,y+b*size)],fill=fill,width=max(1,size*width),capstyle='round')
    oval(.13,.76,.9,.93,'#677867')
    if kind in ('infantry','mech'):
        line(((.45,.64),(.32,.85)),dark,.09); line(((.52,.64),(.67,.85)),dark,.09)
        rect(.32,.4,.6,.69,body); oval(.38,.25,.58,.44,'#efd0a3')
        oval(.32,.17,.62,.35,body); rect(.32,.3,.68,.35,light)
        line(((.5,.5),(.7,.57)),light,.09)
        if kind=='mech':
            rect(.22,.37,.34,.64,dark); line(((.46,.46),(.89,.33)),dark,.11)
        else: line(((.56,.56),(.88,.48)),dark,.06)
    elif kind in ('b_copter','fighter','bomber'):
        if kind=='b_copter':
            line(((.28,.76),(.75,.76)),dark,.035)
            line(((.4,.65),(.37,.76)),dark,.025)
            poly(((.15,.5),(.35,.55),(.51,.37),(.79,.42),(.88,.59),(.66,.7),(.38,.65)),body)
            rect(.69,.46,.82,.56,light)
            line(((.56,.39),(.56,.22)),dark,.04); line(((.13,.22),(.95,.22)),dark,.035)
        else:
            poly(((.1,.39),(.39,.47),(.58,.17),(.68,.17),(.62,.48),(.95,.57),(.63,.66),(.68,.86),(.57,.86),(.4,.65),(.12,.74),(.22,.57)),body)
            line(((.53,.56),(.8,.57)),light,.06)
            if kind=='bomber': rect(.35,.48,.46,.69,dark)
    else:
        if kind=='recon':
            for a in (.25,.66): oval(a,.67,a+.19,.87,dark)
        else:
            rect(.15,.62,.84,.85,dark)
            for a in (.21,.36,.51,.66): oval(a,.69,a+.12,.81,'#8c9990')
        poly(((.15,.62),(.29,.46),(.74,.46),(.89,.65),(.81,.72),(.2,.72)),body)
        rect(.28,.49,.7,.55,light)
        if kind in ('artillery','rocket','anti_air'):
            if kind=='rocket':
                poly(((.33,.42),(.63,.23),(.8,.41),(.5,.58)),body)
                for a in (.48,.57,.66): line(((a,.37),(a+.07,.44)),dark,.04)
            else:
                line(((.5,.48),(.88,.22)),dark,.085)
                line(((.5,.46),(.88,.2)),light,.045)
                if kind=='anti_air': line(((.58,.5),(.95,.25)),light,.045)
        elif kind=='recon':
            rect(.34,.3,.65,.5,body); rect(.54,.34,.66,.44,light)
        else:
            rect(.33,.31,.67,.54,body); rect(.36,.3,.61,.36,light)
            line(((.59,.42),(.96,.42)),dark,.09 if kind=='md_tank' else .065)
    if flash:
        c.create_rectangle(x+.08*size,y+.1*size,x+.97*size,y+.94*size,outline='#fff9ce',width=max(2,size*.04))

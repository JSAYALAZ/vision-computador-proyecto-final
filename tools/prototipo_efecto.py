import cv2, numpy as np, time
from skimage import data
img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
h,w = img.shape[:2]
def stroke_kernel(k):
    c=(k-1)/2; y,x=np.mgrid[0:k,0:k]-c
    u=(x+y)/np.sqrt(2); v=(x-y)/np.sqrt(2)
    su=k/4; sv=max(0.6,k/24)
    K=np.exp(-u**2/(2*su**2)-v**2/(2*sv**2)); return (K/K.sum()).astype(np.float32)
rng=np.random.default_rng(7); grain=rng.integers(0,256,(h,w),dtype=np.uint8)
def patterns(sp):
    y,x=np.mgrid[0:h,0:w]
    p=[((x+y)%sp==0),((x-y)%sp==0),(y%sp==0),(x%sp==0)]
    return [ (q*255).astype(np.uint8) for q in p]
def run(img,k=15,d=9,et=45,sp=9,levels=(175,125,80,40)):
    st={}
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    eq=cv2.equalizeHist(g); st['eq']=eq
    bil=cv2.bilateralFilter(eq,d,60,60); st['bil']=bil
    med=cv2.medianBlur(bil,5)
    lap=cv2.convertScaleAbs(cv2.Laplacian(med,cv2.CV_16S,ksize=3))
    _,edges=cv2.threshold(lap,et,255,cv2.THRESH_BINARY)
    edges=cv2.morphologyEx(edges,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2)))
    edges=cv2.dilate(edges,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    st['edges']=edges
    # trazo: tono + grano, suavizado direccional con kernel propio
    mix=cv2.addWeighted(bil,0.75,grain,0.25,0)
    stroke=cv2.filter2D(mix,-1,stroke_kernel(k)); st['stroke']=stroke
    ink=np.zeros_like(g)
    for thr,pat in zip(levels,patterns(sp)):
        m=cv2.compare(stroke,thr,cv2.CMP_LT)
        ink=cv2.bitwise_or(ink,cv2.bitwise_and(m,pat))
    
    hatch=cv2.bitwise_not(ink); st['hatch']=hatch
    final=cv2.bitwise_and(hatch,cv2.bitwise_not(edges))
    paper=np.array([228,242,250],np.float32); inkc=np.array([70,40,25],np.float32)
    lut=np.array([inkc+(paper-inkc)*v/255 for v in range(256)],np.uint8).reshape(1,256,3)
    out=cv2.LUT(cv2.cvtColor(final,cv2.COLOR_GRAY2BGR),lut); st['final']=out
    return st
st=run(img)
row1=np.hstack([img,cv2.cvtColor(st['bil'],cv2.COLOR_GRAY2BGR),cv2.cvtColor(st['edges'],cv2.COLOR_GRAY2BGR)])
row2=np.hstack([cv2.cvtColor(st['stroke'],cv2.COLOR_GRAY2BGR),cv2.cvtColor(st['hatch'],cv2.COLOR_GRAY2BGR),st['final']])
cv2.imwrite('etapas_efecto.png',np.vstack([row1,row2]))
for k in (3,15,31,61):
    t=time.perf_counter(); run(img,k=k); print(k, round((time.perf_counter()-t)*1000,1),'ms')

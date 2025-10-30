import json, base64, urllib.request
import numpy as np, cv2


def make_test_image():
    img = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (120, 80), (520, 400), (0,0,0), 8)
    cv2.putText(img, 'TEST', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,0), 3)
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY),90])
    if not ok:
        raise RuntimeError('cv2.imencode failed')
    b64 = base64.b64encode(buf).decode('ascii')
    return b64


def post_perspective(b64):
    url = 'http://127.0.0.1:5000/api/perspective'
    data = json.dumps({'image': b64}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode('utf-8')
        print('STATUS', resp.status)
        print(body[:600])


if __name__ == '__main__':
    b64 = make_test_image()
    try:
        post_perspective(b64)
    except Exception as e:
        print('REQUEST ERROR:', e)
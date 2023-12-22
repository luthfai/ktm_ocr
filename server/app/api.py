from app import app, db
from app.models import KartuTandaMahasiswa as KTM
import json
import cv2
import numpy as np
import pytesseract
import time
from flask import request, Response, send_file
import os
@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()



@app.route("/images/<filename>", methods=["GET"])
def get_image(filename):
    # /home/msi/Dev/ml/jadi_mesin/final_project/server/images
    # check if file exist
    if not os.path.isfile("/home/msi/Dev/ml/jadi_mesin/final_project/server/images/" + filename):
        return send_file("/home/msi/Dev/ml/jadi_mesin/final_project/server/images/default.png" , mimetype="image/png")
    return send_file("/home/msi/Dev/ml/jadi_mesin/final_project/server/images/" + filename, mimetype="image/jpg")
        

@app.route("/read", methods=["GET"])
def read():
    data = KTM.query.all()
    data = [{"nim": d.nim, "nama": d.nama, "ttl": d.ttl, "prodi": d.prodi, "alamat1": d.alamat1, "alamat2": d.alamat2, "alamat3": d.alamat3} for d in data]
    return Response(json.dumps(data), status=200, mimetype="application/json")

@app.route("/read", methods=["POST"])
def detect():
    buf = request.files["image_file"]
    nparr = np.fromstring(buf.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    filename = "images/ktm/" + str(time.time()) + ".jpg"
    cv2.imwrite(filename, image)
    image = cv2.imread(filename)
    text = ocr_dong(image)
    print(text)
    if(text == None):
        # return json error no text detected
        return Response(json.dumps({"error": "No text detected"}), status=400, mimetype="application/json")
    else:
        # save to db
        # check if nim already exist
        nim = text[1]
        ktm = KTM.query.filter_by(nim=nim).first()
        if ktm is None:
            # save to db
            ktm = KTM(nim=nim, nama=text[2], ttl=text[3], prodi=text[4], alamat1=text[5], alamat2=text[6], alamat3=text[7])
            db.session.add(ktm)
            db.session.commit()
        # return json
        return Response(json.dumps({
            "type": text[0],
            "nim": text[1],
            "nama": text[2],
            "ttl": text[3],
            "prodi": text[4],
            "alamat1": text[5],
            "alamat2": text[6],
            "alamat3": text[7]
            }), status=200, mimetype="application/json")

@app.route("/update/<nimlama>", methods=["PUT"])
def update(nimlama):
#    no image just update
    ktm = KTM.query.filter_by(nim=nimlama).first()
    if ktm is None:
        return Response(json.dumps({"error": "NIM not found"}), status=400, mimetype="application/json")
    else:
        if nimlama != request.form["nim"]:
            if os.path.isfile("/home/msi/Dev/ml/jadi_mesin/final_project/server/images/" + nimlama + ".jpg"):
                os.rename("/home/msi/Dev/ml/jadi_mesin/final_project/server/images/" + nimlama + ".jpg", "/home/msi/Dev/ml/jadi_mesin/final_project/server/images/" + request.form["nim"] + ".jpg")
        ktm.nim = request.form["nim"]
        ktm.nama = request.form["nama"]
        ktm.ttl = request.form["ttl"]
        ktm.prodi = request.form["prodi"]
        ktm.alamat1 = request.form["alamat1"]
        ktm.alamat2 = request.form["alamat2"]
        ktm.alamat3 = request.form["alamat3"]
        db.session.commit()
        return Response(json.dumps({"success": "Data updated"}), status=200, mimetype="application/json")
    

@app.route("/delete/<nim>", methods=["DELETE"])
def delete(nim):
    ktm = KTM.query.filter_by(nim=nim).first()
    if os.path.isfile("/home/msi/Dev/ml/jadi_mesin/final_project/server/images/" + nim + ".jpg"):
        os.remove("/home/msi/Dev/ml/jadi_mesin/final_project/server/images/" + nim + ".jpg")
    if ktm is None:
        return Response(json.dumps({"error": "NIM not found"}), status=400, mimetype="application/json")
    else:
        db.session.delete(ktm)
        db.session.commit()
        return Response(json.dumps({"success": "Data deleted"}), status=200, mimetype="application/json")
    

# func for ocr
# funsgi 
def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

# approximate the contour by a more primitive polygon shape
def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.032 * peri, True)


def get_contour(contours):
    # loop over the contours
    for c in contours:
        approx = approximate_contour(c)
        # if our approximated contour has four points, we can assume it is receipt's rectangle
        # and the size is not too big
        if len(approx) == 4 and cv2.contourArea(c) < 700000 and cv2.contourArea(c) > 10000:
            return approx

def contour_to_rect(contour):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    # top-left point has the smallest sum
    # bottom-right has the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # compute the difference between the points:
    # the top-right will have the minumum difference 
    # the bottom-left will have the maximum difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def ocr_dong(buf):
    image = buf.copy()
    resize_ratio = 4000 / image.shape[0]
    image = opencv_resize(image, resize_ratio)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # hilangkan noise
    blurred = cv2.GaussianBlur(gray, (51, 51), 0)
    thresh = cv2.threshold(blurred, 170, 255, cv2.THRESH_BINARY)[1]
    edged = cv2.Canny(thresh, 100, 200, apertureSize=3)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    receipt_contour = get_contour(largest_contours)
    if receipt_contour is None:
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        edged = cv2.Canny(thresh, 100, 200, apertureSize=3)
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
        receipt_contour = get_contour(largest_contours)
        if receipt_contour is None:
            return None
    rect = contour_to_rect(receipt_contour)
    pts1 = np.float32(rect)
    # pts2 = np.float32([[375,25],[375,525],[0,525],[0,25]])
    # pts2 = np.float32([[0,525],[0,25],[375,25],[375,525]])
    pts2 = np.float32([[0, 25], [375, 25], [375, 525], [0, 525]])
    # pts2 = np.float32([[100, 400], [475, 400], [475, 900], [100, 900]])
    # pts2 = np.float32([[375, 525], [0, 525], [0, 25], [375, 25]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(image,M,(2000,500))
    text_area = dst[0:800, 500:1800]
    gray_text = cv2.cvtColor(text_area, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray_text, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    gray_text = cv2.GaussianBlur(gray_text, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(gray_text, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 7)
    thresh = 255 - thresh
    opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    cv2.imwrite("images/contour.jpg",dst)

    text = pytesseract.image_to_string(opening, lang='eng')
    text = text.split('\n')
    text = [t for t in text if t]
    print (text)
    text.insert(0, "KTM")
    for t in text:
        if t == " ":
            text.remove(t)
    if len(text) < 8:
        return None
    else:
        cv2.imwrite("images/" + text[1] + ".jpg", dst[25:500, 0:375])
        return text
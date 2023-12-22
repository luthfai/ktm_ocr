from app import db


class KartuTandaMahasiswa(db.Model):
    nim = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(100), nullable=False)
    ttl = db.Column(db.String(100), nullable=False)
    prodi = db.Column(db.String(100), nullable=False)
    alamat1 = db.Column(db.String(100), nullable=False)
    alamat2 = db.Column(db.String(100), nullable=False)
    alamat3 = db.Column(db.String(100), nullable=False)
    date_created = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return "<KTM %r>" % self.nim
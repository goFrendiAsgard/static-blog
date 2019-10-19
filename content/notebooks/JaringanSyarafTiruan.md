---
title: "Neural Network From Scratch"
date: 2019-07-31T07:48:41+07:00
categories:
- Machine Learning
tags:
- Macine Learning
---
# Jaringan Syaraf Tiruan

Ada banyak mitos yang beredar tentang Jaringan Syaraf Tiruan (JST). Misalnya:
* JST mampu membuat sistem yang secerdas manusia.
* JST memberikan akurasi yang lebih tinggi daripada pemrograman konvensional.
* JST dan Kecerdasan Buatan (AI) akan mengambil alih dunia.

Mitos-mitos tersebut sebenarnya cukup menyesatkan. Di satu sisi, hal ini membuat banyak orang menjadi khawatir akan perkembangan AI. Sementara di sisi lain, ada juga mereka yang berharap terlalu banyak pada AI.

Dalam artikel ini, saya ingin menunjukkan bagaimana sebenarnya cara kerja JST secara teknis. Harapannya. Setelah teman-teman membaca artikel ini, teman-teman tidak lagi merasa takut atau berharap terlalu banyak pada AI dan JST.

# Jaringan Syaraf Tiruan VS Jaringan Syaraf Asli

Banyak penulis yang mengawali pembahasannya dengan menganalogikan JST dan jaringan syaraf manusia. Walaupun tidak salah, namun pembahasan tersebut kerap mengarahkan pembaca pada harapan yang berlebihan.

Misalnya begini: "Oh, karena saya bisa jatuh cinta, maka JST pun juga akan bisa jatuh cinta".

Tentu saja tidak benar. Kalau mau jujur, sampai saat ini pun kita tidak terlalu tahu bagaimana otak bekerja. Kita bahkan belum bisa memformulasikan "apa itu kesadaran". Dan JST (sekalipun terinspirasi oleh jaringan syaraf asli), sebenarnya memiliki kapabilitas yang jauh di bawah manusia.

Sebelum kita lanjutkan pada pembahasan yang lebih detail, mari kita mulai dengan permasalahan sederhana.

# Permasalahan Berat Badan Ideal

Anton, Budi, dan Cecep masing-masing mengaku memiliki berat badan ideal. Dari penampakan mereka, tampaknya ketiga orang tersebut tidak berbohong.

Anton memiliki tinggi badan 165 cm dan berat badan 65 kg, sedangkan Budi memiliki tinggi badan 170 cm dan berat badan 70 kg.

Permasalahannya, Si Cecep lupa berapa berat badannya, dan dia hanya ingat tinggi badannya 163 cm. Tanpa memakai timbangan, bisakah kita mengira-ngira berat badan Cecep?

Nah, permasalahan ini sebenarnya bisa diselesaikan dengan aljabar sederhana, selama kita mampu memodelkan persamaannya. Untungnya dalam kasus ini, permodelannya cukup sederhana. (Konon, entah benar atau tidak, permodelan berat badan ideal untuk orang yang tingginya di atas 150 cm itu sesuai dengan fungsi linear).

Misalnya kita anggap berat badan adalah `y`, sedangkan tinggi badan adalah `x`. Maka berat badan ideal bisa ditentukan dengan rumus `y = mx + c` di mana `m` dan `c` adalah angka-angka dengan nilai tertentu.

Oke, tampaknya alam semesta tidak terlalu berpihak pada kita. Karena untuk menentuka berat badan Cecep (`y-cecep`), kita tidak bisa hanya mengandalkan tinggi nya saja (`x-cecep`). Sebaliknya, kita harus tahu juga nilai `m` dan `c` yang ideal.

Untungnya, kita punya Anton dan Budi. Karena rumus berat badan ideal itu seharusnya sama untuk setiap orang, maka nilai `m` dan `c` untuk Anton dan Buid, harusnya berlaku pula untuk Cecep.

Mari kita coba:

```
y = mx + c, di mana y adalah berat badan, sedangkan x adalah tinggi badan.
Maka
   Untuk Anton: 65 = m * 165 + c
   Untuk Budi : 70 = m * 170 + c
```

Mari kita fokus ke Anton dulu:

```
65 = (m * 165) + c
c = 65 - (m * 165)
```

Hmm... tidak terlalu membantu, tapi setidaknya kita tahu bagaimana hubungan antara `c` dan `m`. Sekarang, mari kita fokus ke Budi:

```
70 = m * 170 + c
```

Karena `c = 65 - (m * 165)`, maka kita bisa tulis ulang persamaan berat badan Budi dengan mengganti nilai `c` sehingga menjadi seperti ini:

```
70 = (m * 170) + 65 - (m * 165)
70 - 65 = (m * 170) - (m * 165)
5 = m * (170 * 165)
5 = m * 5
m = 5/5
m = 1
```

Oke, sekarang kita tahu nilai `m`. Mari kita lanjutkan dengan mencari nilai `c`. Kita bisa kembali lagi fokus ke salah satu, Anton atau Budi. Lagi-lagi karena nilai `m` dan `c` untuk kedua persamaan tersebut sama persis, maka tidak masalah memilih Anton atau Budi. Untuk kali ini kita pilih Anton.

```
65 = m * 165 + c
65 = 1 * 165 + c
65 = 165 + c
c = 65 - 165
c = -100
```

Nah, akhirnya kita menemukan nilai `m` dan `c`. Dari sini kita bisa simpulkan bahwa rumus berat badan ideal untuk orang yang tingginya di atas 150 cm adalah `y = (1 * x) - 100`, di mana `y` adalah berat badan, dan `x` adalah tinggi badan.

Akhirnya, dengan rumus berat badan ideal yang sudah kita temukan tadi, kita bisa menemukan berat badan Cecep. Karena tinggi badan Cecep adalah 163 cm, maka:

```
y = (1 * 163) - 100
y = 63
```

Jadi tinggi badan Cecep adalah 63 cm.

Teknik yang baru kita lakukan tadi disebut teknik `subtitusi`. Teknik ini cukup berguna untuk melakukan berbagai macam konversi linear saat kita lupa "rumus resmi" nya. Rumus konversi Celcius-Fahrenhait-Reamur, atau tingkat-kemurnian-emas (dalam persen) ke karat, dan sebagainya. Saat berhadapan dengan kasus-kasus seperti itu, kita hanya butuh kertas dan pensil. Tidak butuh AI :)

Sedikit fakta menarik, ternyata perhitungan kita barusan, sesuai dengan artikel ini: https://lifestyle.kompas.com/read/2017/08/21/081619320/begini-cara-menghitung-berat-badan-ideal-anda. Tentu saja, karena saya sudah mencocokkan angka-angkanya terlebih dahulu :)

# Lebih Jauh dengan Permasalahan Berat Badan Ideal

Berat badan ideal adalah masalah kita semua. Sedikit saja kita kurang bergerak dan terlalu banyak makan, maka kita akan jadi terlalu gemuk. Sebaliknya, jika kita makan terlalu sedikit, maka kita akan jadi terlalu kurus. Dalam hal ini memiliki berat badan ideal itu bisa dianalogikan seperti berjalan di atas seutas tali tipis, sedikit terlalu ke kiri, atau sedikit terlalu ke kanan, maka kita akan jatuh ke dalam lembah `obesitas` atau `mal-nutrisi`.

Nah, menariknya, analogi yang saya tulis barusan tadi ternyata persis sama dengan permodelan matematika yang ada:

```
y = 1 * m - 100
```




```python

```

---
title: "Jaringan Syaraf Tiruan"
date: 2019-07-31T07:48:41+07:00
categories:
- Machine Learning
tags:
- Macine Learning
---
# Mitos

Ada banyak mitos yang beredar tentang Jaringan Syaraf Tiruan (JST). Misalnya:

* JST mampu membuat sistem yang secerdas manusia.
* JST memberikan akurasi yang lebih tinggi daripada pemrograman konvensional.
* JST dan Kecerdasan Buatan (AI) akan mengambil alih dunia.
* JST meningkatkan akurasi sistem.
* JST itu cepat.

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

Berat badan ideal adalah masalah kita semua. Sedikit saja kita kurang bergerak dan terlalu banyak makan, maka kita akan jadi terlalu gemuk. Sebaliknya, jika kita makan terlalu sedikit, maka kita akan jadi terlalu kurus. Dalam hal ini memiliki berat badan ideal itu bisa dianalogikan seperti berjalan di atas seutas tali tipis, sedikit terlalu ke kiri, atau sedikit terlalu ke kanan, maka kita akan jatuh ke dalam lembah `obesitas` (berat badan lebih dari seharusnya) atau `mal-nutrisi` (berat badan kurang dari seharusnya).

Nah, menariknya, analogi yang saya tulis barusan tadi ternyata persis sama dengan permodelan matematika yang ada:

```
y = 1 * m - 100
```

![](http://gofrendiasgard.github.io/images/algebra-line.PNG)

Dengan cara yang sama kita dapat memodelkan `obesitas` dengan pertidaksamaan:

```
y > 1 * m - 100
```

![](http://gofrendiasgard.github.io/images/algebra-greater-than-line.PNG)

Atau `mal-nutrisi` dengan pertidaksamaan:

```
y < 1 * m - 100
```

![](http://gofrendiasgard.github.io/images/algebra-less-than-line.PNG)


# Regresi VS Klasifikasi

Kembali ke kasus Anton, Budi, dan Cecep. Ketiga orang tersebut memiliki data sebagai berikut:

```
Target: y

Nama  | Tinggi (x) | Berat (y)
------------------------------
Anton | 165        | 65
Budi  | 170        | 70
Cecep | 163        | 63
```

Kita cukup beruntung karena jika digambarkan, posisi mereka ada dalam satu garis lurus `y = 1 * x + c`. Di sini kita lihat, bahwa nilai target yang kita cari (dalam hal ini berat), bisa sangat bervariasi. Secara matematis, bahkan kita bisa katakan bahwa kemungkinan nilai berat ini tak terbatas. Bisa jadi ada orang yang beratnya 65,1 kg, atau 65,100013 kg, dan seterusnya.

Nah untuk kasus di mana nilai target yang kita cari memiliki kemungkinan yang sangat banyak, kita sebut permasalahannya sebagai `regresi`. Seperti yang sudah kita duga, `regresi` bisa diselesaikan dengna cara mencari garis/bidang model (dalam kasus kita, model nya adalah garis `y = 1 * x - 100`. Selanjutnya data baru akan kita cocokkan dengan model tersebut sehingga nilai target bisa ditemukan.

Sekarang bayangkan, bagaimana jika kita punya kasus yang berbeda

```
Target: z

Nama   | Tinggi (x) | Berat (y) | Kategori (z)
-----------------------------------------------
Anton  | 165        | 65        | Ideal
Budi   | 170        | 70        | Ideal
Cecep  | 163        | 63        | Ideal
Didit  | 163        | 40        | Mal-nutrisi
Emil   | 170        | 50        | Mal-nutrisi
Frank  | 150        | 40        | Mal-nutrisi
Ganot  | 170        | 100       | Obesitas
Herman | 165        | 80        | Obesitas
Ical   | 180        | 120       | Obesitas
```

Kali ini target kita adalah kategori. Berbeda dengan kasus `regresi` di contoh sebelumnya, sekarang kita hanya punya tiga kemungkinan target, yakni `ideal`, `mal-nutrisi`, dan `obesitas`. Permasalahan ini disebut permasalahan `klasifikasi`.

Jika pada `regresi` kita mencocokkan data terhadap model, maka pada `klasifikasi`, kita menggunakan model untuk memisahkan data (Mirip seperti kasus `pertidaksamaan linear` pada contoh sebelumnya).

Walaupun sepintas permasalahan regresi dan klasifikasi tampak berbeda, namun langkah awalnya sama: Menentukan model.

Dengan sedikit abstraksi, kita bisa mengubah permasalahan klasifikasi ke ranah regresi. Kita ambil kembali contoh di atas. Kali ini dengan atribut `rasio`:

```
Target: z

Nama   | Tinggi (x) | Berat (y) | Rasio (w=y/(x-100)) | Kategori (z)
--------------------------------------------------------------------
Anton  | 165        | 65        | 1                   | Ideal
Budi   | 170        | 70        | 1                   | Ideal
Cecep  | 163        | 63        | 1                   | Ideal
Didit  | 163        | 40        | 0.63                | Mal-nutrisi
Emil   | 170        | 50        | 0.71                | Mal-nutrisi
Frank  | 150        | 40        | 0.80                | Mal-nutrisi
Ganot  | 170        | 100       | 1.42                | Obesitas
Herman | 165        | 80        | 1.23                | Obesitas
Ical   | 180        | 120       | 1.5                 | Obesitas
```

Pertama-tama kita melakukan `regresi` untuk mencari nilai rasio, kemudian kita melakukan klasifikasi berdasarkan rasio tersebut. Misalnya, jika rasio kurang dari satu berarti mal-nutrisi, jika lebih dari satu, berarti obesitas, dan jika sama dengan satu berarti ideal.

Abstraksi tersebut bukanlah satu-satunya cara yang mungkin. Masih ada banyak cara lain yang tidak kita bahas di sini.


# Kita dan Alam Semesta yang Tidak Ideal

Regresi adalah satu teknik yang sangat berguna. Dalam dunia yang ideal, kita bisa saja memprediksi harga saham dan tanah ataupun menebak perasaan 'si dia' dengan menggunakan permodelan yang tepat.

Sayangnya, __alam semesta tidak ideal__, dan __kita tidak maha tahu__. Ketidak-idealan alam semesta mungkin bisa digambarkan seperti berikut:

![](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/1200px-Linear_regression.svg.png)

Pada gambar di atas, kita bisa lihat bahwa dari sekian banyak titik yang ada,jarang sekali yang benar-benar berada dalam garis model. Banyak hal di dunia ini yang tidak bisa 100% ditebak. Lemparan dua mata dadu misalnya, tidak mungkin memunculkan angka 13 atau 0. Kemungkinan kecil memunculkan angka 1 atau 12, dan besar kemungkinan memunculkan angka 5,6, dan 7. Hanya sejauh itulah yang kita tahu.

Ada berbagai model statistik untuk menghitung peluang, tapi tidak ada yang 100% akurat. Maka dalam dunia yang tidak ideal ini, perlu sekali bagi kita untuk berkompromi dan men-toleransi banyak hal.

Salah satu cara untuk menerapkan `tolearansi` pada matematika adalah dengan membuat `error function` atau `loss function`. Tujuan kita pun bergeser, bukan lagi menebak secara akurat, namun menebak dengan kesalahan seminimal mungkin. Salah satu `error function` yang paling primitif dan cukup sering dipakai adalah:

$$E=\sqrt{(Target-Output)^2}$$

Kalian mungkin bertanya-tanya, kenapa tidak sekedar `Target-Output` saja? Nah, `Target-Output` mungkin saja memberikan hasil negatif. Ini akan menimbulkan masalah saat kita mencoba menjumlahkan nilai `error` total.

```
Target | Output | Target - Output | sqrt((Target-Output)^2) 
-----------------------------------------------------------
 8     | -8     |  16             | 16
-8     |  8     | -16             | 16
-----------------------------------------------------------
Total           |   0 (Loh???)    | 32 (Seharusnya begini)
```

Adanya `akar` dan `kuadrat` pada rumus `error function` dimaksudkan untuk menghindari masalah ini.

Dengan menentukan `error function` yang tepat, maka kita mampu __mengukur kesalahan__ dan __membuat model yang lebih toleran__.

Tapi kita masih punya masalah kedua. Kita tidak maha tahu, dan tidak semua masalah bisa diselesaikan dengan persamaan linear.

Lalu apa yang harus kita lakukan?

Jawaban yang naif adalah "mencoba satu-satu". Dalam beberapa kasus, kita bisa saja cukup beruntung dan menemukan model yang tepat.

Dalam keadaan normal, biasanya pola sebaran statistika akan mengikuti fungsi Gauss (https://en.wikipedia.org/wiki/Gaussian_function). Tapi tentu saja, ini pertaruhan yang belum tentu berhasil.

Cara yang sedikit lebih baik, adalah dengan membuat kerangka model yang cukup general. Bisa diumpamakan seperti kita memakai swiss-army knife yang jika dilipat-lipat dengan konfigurasi yang benar, bisa memunculkan suatu alat yang berguna untuk menyelesaikan permasalahan kita.

Menurut saya, inilah definisi jaringan syaraf tiruan: Sebuah model matematis yang cukup general, mudah dikonfigurasi dan diotak-atik. Tapi cukup susah untuk dihitung secara manual. :)

# Fondasi Jaringan Syaraf Tiruan: Perceptron

Seperti yang biasa digambarkan di buku-buku atau artikel yang membahas tentang JST, struktur umum JST adalah sebagai berikut:

![](https://upload.wikimedia.org/wikipedia/commons/9/99/Neural_network_example.svg)

Setiap lingkaran pada gambar di atas disebut `perceptron` atau `neuron`. Kita bisa menambahkan jumlah neuron dan menatanya sesuai kebutuhan.

Jika dibedah lebih detail, maka setiap lingkaran atau `perceptron` pada gambar sebelumnya bisa dijabarkan sebagai berikut:

![](https://upload.wikimedia.org/wikipedia/commons/3/31/Perceptron.svg)

Saat perceptron-perceptron tersebut dirangkai, maka akan muncul model matematika yang lebih kompleks. Sebelum masuk ke sana, ada baiknya kita membedah dan memahami bagaimana `perceptron` bekerja.

## Feed Forward

Mari kita lihat lagi persamaan yang ada pada sebuah perceptron:

$$o=f(\Sigma^n_{k=1}i_k.W_k)$$

Kita bisa memecah persamaan tersebut menjadi dua bagian, yakni

$$y=\Sigma^n_{k=1}i_k.W_k ... (penjumlahan)$$

dan

$$o=f(y) ... (aktivasi)$$

### Bagian Penjumlahan

Oke, mari kita fokus terlebih dahulu pada bagian pertama. Bagian ini sebenarnya adalah generalisasi dari persamaan linear yang sudah kita bahas pada kasus `berat badan ideal`. Jika dijabarkan lagi, kita akan memperoleh bentuk penjumlahan sebagai berikut:

$$y=\Sigma^n_{k=1}(i_k.W_k)$$

$$y=i_1.W_1 + i_2.W_2 + ... + i_n.W_n$$

Ini mirip sekali dengan persamaan garis linear:

$$y=m.x + c$$

Bayangkan saja, sekarang untuk menetukan `y` (berat badan), kita tak lagi hanya bergantung `x` (tinggi badan), melainkan ada faktor-faktor lain seperti `x2` (ukuran sepatu), `x3` (cangkir kopi yang dihabiskan tiap hari), `x4` (Porsi nasi sekali makan) dan seterusnya. Maka kita akan menemukan persamaan seperti ini:

$$y=m_1.x_1 + m_2.x_2 + m_3.x_3 + m_4.x_4 + c$$

Persamaan tersebut sudah cukup mirip dengan persamaan `bagian penjumlahan` pada `perceptron`. Apalagi jika kita mengganti `m` dengan `W` dan `x` dengan `i`.

Supaya lebih mirip lagi, biasanya pada sebuah perceptron ditambahkan nilai `bias`. Beberapa referensi menggunakan simbol tersendiri untuk nilai `bias`. Tapi di tulisan ini, kita akan simbolkan bias dengan `m_5`.

$$y=m_1.x_1 + m_2.x_2 + m_3.x_3 + m_4.x_4 + m_5.x_5$$

dimana `x_5` selalu bernilai `1`.

Sekarang kita bisa yakin bahwa `bagian penjumlahan` pada sebuah `perceptron` tak lain adalah __persamaan linear__.

Kembali lagi ke kasus awal (dengan sedikit modifikasi):

```
Kategori:
  0   : mal-nutrisi
  0.5 : ideal
  1   : obesitas

Nama  | Tinggi (i_1) | Porsi makan (i_2) | Kategori (t)
-------------------------------------------------------
Anton | 165          | 3                 | 0.5 (ideal)
```

Maka kita bisa menghitung nilai `y` dengan cara berikut:

$$y=\Sigma^n_{k=1}(i_k.W_k)$$

$$y=i_1.W_1 + i_2.W_2 + i_3.W_3$$

Nilai `w1`, `w2`, dan `w3` akan kita tentukan secara acak. Misalnya kita ambil `w1` = 0.3, `w2` = 0.2, dan `w3` = 0.1, maka:

```
y = 165 * 0.3 + 3 * 0.2 + 0.1
```

### Bagian Aktivasi

Usai melakukan penjumlahan, langkah selanjutnya adalah melakukan aktivasi. Jika pada bagian penjumlahan kita sudah menciptakan satu persamaan linear, maka fungsi aktivasi bertugas untuk menghadapi ketidak-linearan data. Pada dasarnya fungsi aktivasi bertujuan untuk memetakan output persamaan linear dari `bagian penjumlahan` menjadi tidak linear.

Ada beberapa fungsi aktivasi yang umum dipakai. Penjelasan lebih lanjut tentang fungsi aktivasi bisa dilihat pada artikel ini: https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

![](https://miro.medium.com/max/1005/1*p_hyqAtyI8pbt2kEl6siOQ.png)

Dari sekian banyak fungsi aktivasi yang ada, tampaknya ada dua yang paling sering dipakai, yakni `sigmoid` ( atau `logistic`) dan `ReLu` (serta `Leaky ReLu`).

Jika kita yakin data kita pasti linear, maka ada baiknya kita tidak memakai jaringan syaraf tiruan. Lebih baik lakukan `linear regression` yang jauh lebih sederhana dan cepat. Atau kalau masih ingin memakai JST, gunakan fungsi aktivasi linear (yang pada dasarnya tidak melakukan apapun).

Untuk menghindari `vanishing gradient problem`, umumnya praktisi deep learning lebih menyukai `ReLu` atau `Leaky ReLu`. Kedua fungsi ini juga kerap dipilih karena perhitungannya yang sederhana.

Ok, kembali ke kasus awal. Semisal kita memilih fungsi `logistic`, maka kita harus melakukan ini:

```
y = 165 * 0.3 + 3 * 0.2 + 0.1

o = f(y)
o = 1 / [1 + e^-(165 * 0.3 + 3 * 0.2 + 0.1)]
```

`e` sendiri adalah bilangan euler yang nilainya mendekati `2.718` (https://en.wikipedia.org/wiki/E_(mathematical_constant))


```python
import math
1/(1 + math.exp(165 * 0.3 + 3 * 0.2 + 0.1))
```




    1.5791268155225368e-22



## Back Propagation

Jika diperhatikan, tentu saja hasil perhitungan feed-forward ini meleset jauh. Tapi jangan khawatir. JST punya mekanisme untuk memperbaiki nilai `W` sehingga nantinya nilai output yang dihasilkan akan mendekati target.

Untuk memperbaiki nilai `W`, maka kita butuh dua macam informasi:

* Seberapa besar nilai `W`
* Bagaimana kita harus mengubah nilai `W`

JST menjawab pertanyaan pertama dengan menyediakan satu parameter yang disebut `learning rate`.

Untuk pertanyaan kedua, jawabannya ternyata jauh lebih kompleks, karena kita membutuhkan perhitungan kalkulus. Sebelum kita melangkah lebih jauh, mari kita lihat, bagaimana kalkulus bisa membantu.


```python

```

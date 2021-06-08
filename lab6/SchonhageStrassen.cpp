#include <cmath>
#include <complex>
#include <vector>
#include <string> 


#include "SchonhageStrassen.h"

using namespace std;

/**
 * @brief Переставить n первых бит в числе a в конец
 * @param a число для переворота
 * @param n кол-во бит
 * @return перевёрнутое число
 */
static int reverseBits(int a, int n) {
  int res = 0;
  for (int i = 0; i < n; i++) {
    res <<= 1;
    res |= a & 1;
    a >>= 1;
  }
  return res;
}

/**
 * @brief Подсчитать натуральный логарифм
 * @param a число для подсчёта
 * @return кол-во бит
 */
static int getLogN(int a) {
  int cnt = 0;
  do {
    cnt++;
    a >>= 1;
  } while (a);
  return cnt;
}

/**
 * @brief Развернуть числа в векторе а, переместить биты в каждом числе вектора на другой конец
 * @param a вектор чисел
 */
static void swapAndReverseBits(vector<complex<double>> &vec) {
  const int vecSize = vec.size();
  int len = getLogN(vecSize) - 1;
  vector<complex<double>> res(vecSize);
  for (int i = 0; i < vecSize; i++)
    res[i] = vec[reverseBits(i, len)];
  vec = res;
  vec.shrink_to_fit();
}

/**
 * @brief Алгоритм быстрого преобразования Фурье 
 * @ref https://en.wikipedia.org/wiki/Fast_Fourier_transform
 * @ref http://e-maxx.ru/algo/export_fft_multiply
 */
static void fastFourierTransform(vector<complex<double>> &vec, bool back) {
  const double pi = acos(-1);

  swapAndReverseBits(vec);
  int n = vec.size();
  double t = (back ? -1.0f : 1.0f);

  for (int m = 2; m <= n; m *= 2) {
    complex<double> wm(cos(t * 2 * pi / (double)m), sin(t * 2 * pi / (double)m));

    for (int k = 0; k < n; k += m) {
      complex<double> w(1);
      for (int j = 0; j < m / 2; ++j) {
        complex<double> a0 = vec[k + j];
        complex<double> w_a1 = w * vec[k + j + m / 2];
        vec[k + j] = a0 + w_a1;
        vec[k + j + m / 2] = a0 - w_a1;

        if (back) {
          vec[k + j] /= 2.0f;
          vec[k + j + m / 2] /= 2.0f;
        }
        w *= wm;
      }
    }
  }
}

/**
 * @brief Возвести число в ближайшую степень двойки
 * @param a число для возведения в степень
 * @return ближайшее число, иначе -1
 */
static int toPow2(int a) {
  for (int i = 0; i < 32; ++i)
    if ((1 << i) >= a)
      return 1 << i;
  return -1;
}

lNum SS_fastMultiply(lNum num1, lNum num2) {
  int n = toPow2(max(num1.size(), num2.size())) * 2;
  num1.resize(n);
  num2.resize(n);

  vector<complex<double>> ac(num1.begin(), num1.end());
  vector<complex<double>> bc(num2.begin(), num2.end());

  fastFourierTransform(ac, false);
  fastFourierTransform(bc, false);

  vector<complex<double>> cc(n);
  for (int i = 0; i < n; ++i)
    cc[i] = ac[i] * bc[i];
  fastFourierTransform(cc, true);

  lNum c(n, 0);
  for (int i = 0; i < n; i++)
    c[i] += (int)(cc[i].real() + 0.5);

  for (int i = 0; i < n - 1; i++) {
    if (c[i] > 9) {
      c[i + 1] += c[i] / 10;
      c[i] %= 10;
    }
  }

  while (c[c.size() - 1] > 9) {
    c.push_back(c[c.size() - 1] / 10);
    c[c.size() - 2] %= 10;
  }

  while (c.back() == 0 && c.size() > 1)
    c.pop_back();

  c.shrink_to_fit();
  return c;
}

lNum SS_intArray2LNum(int* array, int size) {
  lNum num;
   for (int i = 0; i < size; i++)
      num.push_back(array[i]);
  return num;
}

int *SS_lNum2Array(lNum num, int* size) {
  int* res = new int[num.size()];
  for (int i = 0; i < num.size(); i++)
    res[i] = num[i];
  if (size) 
    *size = num.size();
  return res;
}

string SS_lNum2String(lNum num) {
  string s = "";
  for (int i = num.size() - 1; i >= 0; i--)
    s.append(to_string(num[i]));
  return s;
}



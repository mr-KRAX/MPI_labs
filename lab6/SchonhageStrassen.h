/**
 * @file SchonhageStrassen.h
 * @author Alexandr
 * @brief Модуль реализующий умножение длинных чисел методом Шёнхаге - Штрассена 
 * (алгоритм умножения больших целых чисел, основанный на быстром преобразовании Фурье)
 * @version 0.1
 * @date 2021-06-08
 * 
 * @copyright Copyright (c) 2021
 */

#pragma once
#include <vector>

typedef std::vector<int> lNum;

/**
 * @brief Перемножить два длинных целых числа
 * @param num1 первое длинное число
 * @param num2 второе длинное число
 * @result результат
 */
lNum SS_fastMultiply(lNum num1, lNum num2);

lNum SS_intArray2LNum(int* array, int size);

int* SS_lNum2Array(lNum num, int *size = nullptr);


std::string SS_lNum2String(lNum num); 
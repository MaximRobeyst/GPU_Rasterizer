#include "Triangle.h"
#include "const.h"

#include <algorithm>

float Clamp(float value, float min, float max)
{
	if (value < min)
		return min;
	if (value > max)
		return max;

	return value;
}
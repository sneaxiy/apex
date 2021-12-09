#include <iostream>
#include "fmhalib.h"

void PrintError() {
  const char *err = fmhalib_error();
  if (err) std::cout << err << std::endl;
}

void PrintErrorCxx() {
  const char *err = fmhalib::error();
  if (err) std::cout << err << std::endl;
}

int main() {
  fmhalib_fwd(nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, true, 0, nullptr, 0, false, nullptr, nullptr, nullptr);
  PrintError();

  fmhalib_fwd_nl(nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, true, 0, nullptr, 0, false, nullptr, nullptr, nullptr);
  PrintError();

  fmhalib_bwd(nullptr, nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, nullptr, nullptr, nullptr); 
  PrintError();

  fmhalib_bwd_nl(nullptr, nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, nullptr, nullptr, nullptr, nullptr);
  PrintError();

  // The following codes show the dynload style
  fmhalib::fwd(nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, true, 0, nullptr, 0, false, nullptr, nullptr, nullptr);
  PrintErrorCxx();

  fmhalib::fwd_nl(nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, true, 0, nullptr, 0, false, nullptr, nullptr, nullptr);
  PrintErrorCxx();

  fmhalib::bwd(nullptr, nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, nullptr, nullptr, nullptr);
  PrintErrorCxx();

  fmhalib::bwd_nl(nullptr, nullptr, nullptr, 0, 0, 0, 0, 0.0f, 0, nullptr, nullptr, nullptr, nullptr);
  PrintErrorCxx();
  
  return 0;
}

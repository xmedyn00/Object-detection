#include "BCproject.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    BCproject w;
    w.show();
    return a.exec();
}

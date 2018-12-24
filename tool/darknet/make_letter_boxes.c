/*
 * Copyright (C) 2018  Zhijin Li
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
** ---------------------------------------------------------------------------
**
** File: make_darknet_letter_boxes.c for Cam-Vision
**
** Created by Zhijin Li
** E-mail:   <jonathan.zj.lee@gmail.com>
**
** Started on  Sun Dec 23 23:11:49 2018 Zhijin Li
** Last update Mon Dec 24 00:53:06 2018 Zhijin Li
** ---------------------------------------------------------------------------
*/


# include "darknet.h"
# include "dirent.h"


/*
 * @brief Print program usage on console.
 */
void show_usage()
{
  printf("Usage:\nmake_letter_boxes\n");
  printf("[path/image/input/output]\n");
  printf("[input extension (e.g. jpg)]\n");
  printf("<optional: box width (defaults to 416)>\n");
  printf("<optional: box height (defaults to 416)>\n");
}


/*
 * @brief Read an image and applies letter box transform.
 *
 * @param fname: filename.
 * @param width: letter box width.
 * @param height: letter box height.
 * @return Letter box transformed image.
 *
 */
image make_letter_box(char *fname, int width, int height)
{
  image img = load_image_color(fname, 0, 0);
  image letter_box = letterbox_image(img, width, height);
  return letter_box;
}


/*
 * @brief Process an image.
 *
 * Applies letter box transform and save image as binary file.
 *
 * @param fname: filename.
 * @param width: letter box width.
 * @param height: letter box height.
 *
 */
void process_img(char *fname, int width, int height)
{
  image __ltb = make_letter_box(fname, width, height);
  char savename[1024];
  sprintf(savename, "%s.ltb.raw", fname);

  FILE *__f = fopen(savename, "wb");
  fwrite(__ltb.data, sizeof(float), __ltb.h*__ltb.w*__ltb.c, __f);
}


/*
 * @brief Process all images inside a directory.
 *
 * Applies letter box transform and save images as binary file.
 *
 * @param dirname: directory name.
 * @param ext: image file extension.
 * @param width: letter box width.
 * @param height: letter box height.
 *
 */
void process_all(char *dirname, char *ext, int width, int height)
{
  DIR *directory = opendir(dirname);
  struct dirent *elem;

  if (directory != NULL) {
    while( (elem = readdir(directory)) )
    {
      char fname[2048];
      sprintf(fname, "%s/%s", dirname, elem->d_name);
      if( strcmp(fname+strlen(fname)-strlen(ext), ext) == 0 )
      {
        printf("letter boxing: %s\n", fname);
        process_img(fname, width, height);
      }
    }
  }
  closedir(directory);
}


int main(int argc, char **argv)
{

  if( argc != 3 && argc != 5 )
  {
    show_usage();
    return 1;
  }

  char *ext = argv[2];
  int width  = 416;
  int height = 416;

  if( argc == 5 )
  {
    width  = atoi(argv[3]);
    height = atoi(argv[4]);
  }

  printf("letter box width: %d, height: %d\n", width, height);
  process_all(argv[1], ext, width, height);
  printf("letter box finished\n");

  return 0;
}

var gulp = require('gulp')
var gutil = require('gulp-util')
var coffee = require('gulp-coffee')
var concat = require('gulp-concat')

gulp.task('build', function () {
  gulp.src('src/*.coffee')
    .pipe(concat('MMD.coffee'))
    .pipe(gulp.dest('.'))
    .pipe(coffee({bare: true}).on('error', gutil.log))
    .pipe(gulp.dest('.'))
})

gulp.task('watch', function () {
  gulp.watch('src/*.coffee', ['build'])
})

gulp.task('default', ['build'])
